import os
import sys
import socket
import threading
import torch
from tkinter import filedialog, messagebox, Menu
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='librosa.*')
warnings.filterwarnings("ignore", category=FutureWarning, module='librosa.*')

# ================================================================
# SINGLE-INSTANCE PROTECTION
# Предотвращает открытие двух копий (актуально при запуске из терминала на macOS)
# ================================================================
_LOCK_PORT = 47182  # произвольный свободный порт

def _is_already_running() -> bool:
    """Пытается занять порт. Если не получилось — другой экземпляр уже запущен."""
    global _lock_socket
    try:
        _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _lock_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        _lock_socket.bind(("127.0.0.1", _LOCK_PORT))
        _lock_socket.listen(1)
        return False  # порт свободен — мы первые
    except OSError:
        return True   # порт занят — уже запущен

if _is_already_running():
    print("[QazTriber] Уже запущен, завершаем второй экземпляр.")
    sys.exit(0)

# --- ГЛОБАЛЬНЫЙ ПАТЧ PYTORCH (для совместимости с v2.6+) ---
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs or kwargs['weights_only'] is True:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import tkinter as tk
import librosa
import numpy as np
import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
import datetime
import time
import webbrowser

from engine.whisper_engine import WhisperEngine

DEFAULT_PROMPT = "Сәлем, қалайсың? Жақсы. Пойдем сегодня в кафе, там обсудим проект."


# ================================================================
# WAVEFORM WIDGET
# ================================================================
class WaveformWidget:
    CANVAS_H = 90
    BAR_LOW  = "#00cc55"
    BAR_MID  = "#ffaa00"
    BAR_HIGH = "#ff3300"
    WAVE_COL = "#00aaff"
    SEL_COL  = "#003366"
    MARK_L   = "#00ff88"
    MARK_R   = "#ff5500"
    PROG_COL = "#ffffff"  # Цвет линии воспроизведения

    def __init__(self, parent, on_crop, on_seek):
        self._on_crop = on_crop
        self._on_seek = on_seek
        self.audio_data = None
        self.sr = 16000
        
        self.view_start = 0
        self.view_end = 0

        self._live_rms = []
        self._MAX_LIVE = 120
        self.sel_start = None
        self.sel_end   = None
        self._drag_x0  = None
        self.play_progress = 0.0

        self.frame = ctk.CTkFrame(parent, fg_color="#111111", corner_radius=8)
        self.canvas = tk.Canvas(
            self.frame, bg="#111111", height=self.CANVAS_H,
            highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill="x")
        self.btn_scissors = ctk.CTkButton(
            self.frame, text="✂️ Обрезать выделенное", height=28,
            fg_color="#333", hover_color="#555",
            font=ctk.CTkFont(size=11), command=self._do_crop
        )
        self.canvas.bind("<ButtonPress-1>",  self._press)
        self.canvas.bind("<B1-Motion>",       self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Configure>",       lambda e: self._redraw())

        # Масштабирование и прокрутка (Mac / Windows)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Command-MouseWheel>", self._on_zoom)
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)

    def pack(self, **kw): self.frame.pack(**kw)
    def grid(self, **kw): self.frame.grid(**kw)

    def push_live(self, chunk: np.ndarray):
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        self._live_rms.append(rms)
        if len(self._live_rms) > self._MAX_LIVE:
            self._live_rms.pop(0)
        self._draw_live()
            
    def _draw_live(self):
        c = self.canvas
        w = c.winfo_width(); h = self.CANVAS_H
        if w <= 1 or not self._live_rms: return
        c.delete("all")
        n = len(self._live_rms)
        bw = max(2, w // max(n, 1))
        peak = max(max(self._live_rms), 1e-5)
        mid = h // 2
        for i, rms in enumerate(self._live_rms):
            norm = rms / peak
            bh = max(1, int(norm * mid * 0.9))
            x = i * bw
            col = self.BAR_HIGH if norm > 0.8 else self.BAR_MID if norm > 0.5 else self.BAR_LOW
            c.create_rectangle(x, mid - bh, x + bw - 1, mid + bh, fill=col, outline="")

    def show_waveform(self, audio: np.ndarray, sr: int = 16000):
        self.audio_data = audio; self.sr = sr
        self.view_start = 0
        self.view_end = len(audio)
        self.sel_start = self.sel_end = None
        self.btn_scissors.pack_forget()
        self._redraw()

    def clear(self):
        self.audio_data = None; self._live_rms = []
        self.sel_start = self.sel_end = None
        self.view_start = self.view_end = 0
        self.btn_scissors.pack_forget()
        self.canvas.delete("all")

    def _on_scroll(self, e):
        # Панорамирование
        if self.audio_data is None: return
        total_n = len(self.audio_data)
        n_view = self.view_end - self.view_start
        shift = -e.delta * (n_view * 0.005) # на маке delta большая (+/- 10..100)
        
        new_start = self.view_start + shift
        new_end = self.view_end + shift
        
        if new_start < 0:
            new_start = 0
            new_end = n_view
        if new_end > total_n:
            new_end = total_n
            new_start = total_n - n_view
            
        self.view_start, self.view_end = new_start, new_end
        self._redraw()

    def _on_zoom(self, e):
        if self.audio_data is None: return
        w = max(1, self.canvas.winfo_width())
        # Точка зума (процент от ширины экрана)
        focus = e.x / w
        n_view = self.view_end - self.view_start
        
        # e.delta положительный = zoom in
        factor = 0.8 if e.delta > 0 else 1.25
        
        new_n_view = n_view * factor
        # Ограничения зума
        min_samples = int(self.sr * 0.5) # Максимум зум до 0.5 секунд
        if new_n_view < min_samples: new_n_view = min_samples
        if new_n_view > len(self.audio_data): new_n_view = len(self.audio_data)
        
        # Расширяем/сужаем вокруг точки фокуса
        new_start = self.view_start + (n_view - new_n_view) * focus
        new_end = new_start + new_n_view
        
        if new_start < 0:
            new_start = 0
            new_end = new_n_view
        if new_end > len(self.audio_data):
            new_end = len(self.audio_data)
            new_start = new_end - new_n_view
            
        self.view_start, self.view_end = new_start, new_end
        self._redraw()

    def _sample_to_x(self, s: float) -> int:
        n_view = self.view_end - self.view_start
        if n_view <= 0: return -1
        w = self.canvas.winfo_width()
        return int((s - self.view_start) / n_view * w)

    def _x_to_sample(self, x: int) -> float:
        w = max(1, self.canvas.winfo_width())
        n_view = self.view_end - self.view_start
        return self.view_start + (x / w) * n_view

    def _redraw(self):
        if self.audio_data is None:
            self._draw_live(); return
        c = self.canvas
        w = c.winfo_width(); h = self.CANVAS_H
        if w <= 1: return
        c.delete("all")
        
        vs = int(self.view_start)
        ve = int(self.view_end)
        if ve <= vs: return
        
        # Выделение
        if self.sel_start is not None and self.sel_end is not None:
            s1, s2 = sorted([self.sel_start, self.sel_end])
            if s2 > vs and s1 < ve:
                x1 = self._sample_to_x(max(s1, vs))
                x2 = self._sample_to_x(min(s2, ve))
                c.create_rectangle(x1, 0, x2, h, fill=self.SEL_COL, outline="")
                
        mid = h // 2
        
        # Оптимизированный рендер
        view_audio = self.audio_data[vs:ve]
        n = len(view_audio)
        step = max(1, n // w)
        
        # Быстрое нахождение пиков (вместо медленного for-if)
        render_w = min(w, n // step)
        if render_w > 0:
            # Обрезаем так, чтобы длина делилась на step
            trimmed = view_audio[:render_w * step]
            reshaped = trimmed.reshape(-1, step)
            
            # Супер-оптимизация для огромных файлов (2-х часовых)
            if step > 1000:
                peaks = np.max(np.abs(reshaped[:, ::(step//100)]), axis=1) * (mid * 0.9)
            else:
                peaks = np.max(np.abs(reshaped), axis=1) * (mid * 0.9)
            
            for x, peak in enumerate(peaks):
                if peak > 0.5:
                    bh = int(peak)
                    c.create_line(x, mid - bh, x, mid + bh, fill=self.WAVE_COL, width=1)

        # Рисуем ползунок прогресса если он на экране
        prog_sample = self.play_progress * len(self.audio_data)
        if vs <= prog_sample <= ve:
            px = self._sample_to_x(prog_sample)
            c.create_line(px, 0, px, h, fill=self.PROG_COL, width=2)
            
        # Маркеры выделения
        if self.sel_start is not None and vs <= self.sel_start <= ve:
            c.create_line(self._sample_to_x(self.sel_start), 0, self._sample_to_x(self.sel_start), h, fill=self.MARK_L, width=2)
        if self.sel_end is not None and vs <= self.sel_end <= ve:
            c.create_line(self._sample_to_x(self.sel_end), 0, self._sample_to_x(self.sel_end), h, fill=self.MARK_R, width=2)

    def _press(self, e):
        if self.audio_data is None: return
        self._drag_x0 = e.x
        self.sel_start = self._x_to_sample(e.x); self.sel_end = None
        self.btn_scissors.pack_forget(); self._redraw()

    def _drag(self, e):
        if self.audio_data is None: return
        self.sel_end = self._x_to_sample(e.x); self._redraw()

    def _release(self, e):
        if self.audio_data is None: return
        self.sel_end = self._x_to_sample(e.x)
        
        if self._drag_x0 is not None and abs(e.x - self._drag_x0) < 5:
            # Это клик -> seek
            self.sel_start = self.sel_end = None
            self.btn_scissors.pack_forget()
            s = self._x_to_sample(e.x)
            frac = max(0.0, min(1.0, s / len(self.audio_data)))
            self._on_seek(frac)
        else:
            if self.sel_start is not None and self.sel_end is not None:
                s, en = sorted([self.sel_start, self.sel_end])
                if en - s > self.sr * 0.1:
                    self.sel_start, self.sel_end = s, en
                    self.btn_scissors.pack(fill="x", padx=6, pady=(2, 6))
                else:
                    self.sel_start = self.sel_end = None
                    self.btn_scissors.pack_forget()
        self._redraw()

    def set_progress(self, frac: float):
        self.play_progress = max(0.0, min(1.0, frac))
        
        # Автоматическая прокрутка (Auto-scroll) если курсор уходит за правую половину экрана
        if self.audio_data is not None:
            prog_sample = self.play_progress * len(self.audio_data)
            n_view = self.view_end - self.view_start
            
            # Если курсор приблизился к правому краю (80%) или вышел за него
            if prog_sample > self.view_start + n_view * 0.8:
                new_start = prog_sample - n_view * 0.2
                new_end = new_start + n_view
                if new_end > len(self.audio_data):
                    new_end = len(self.audio_data)
                    new_start = new_end - n_view
                self.view_start, self.view_end = new_start, new_end
            # Если курсор левее экрана
            elif prog_sample < self.view_start:
                new_start = prog_sample - n_view * 0.1
                if new_start < 0: new_start = 0
                self.view_start, self.view_end = new_start, new_start + n_view
                
        self._redraw()

    def _do_crop(self):
        if self.audio_data is None or self.sel_start is None or self.sel_end is None: return
        s = int(min(self.sel_start, self.sel_end))
        e = int(max(self.sel_start, self.sel_end))
        self._on_crop(self.audio_data[s:e], self.sr)


class QazTriberApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("QazTriber — Умный транскрибатор")
        self.geometry("1080x800")
        self.configure(fg_color="#121212")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Системное меню (macOS Top Menu)
        menu_bar = Menu(self)
        help_menu = Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Обратная связь (Telegram Написать)", 
                              command=lambda: webbrowser.open("https://t.me/nurlan_nursultan"))
        menu_bar.add_cascade(label="Помощь", menu=help_menu)
        self.config(menu=menu_bar)

        # --- СОСТОЯНИЯ ---
        self.engine: Optional[WhisperEngine] = None
        self.audio_path: Optional[str] = None
        self.model_path: Optional[str] = None
        self.is_recording: bool = False
        self.recording_data: list = []
        self.stream: Optional[sd.InputStream] = None
        self.stop_pulsing: bool = False
        self.stop_event = threading.Event()  # жёсткая остановка через Event

        # --- ПЛЕЕР ---
        self.is_playing: bool = False
        self.playback_data: Optional[np.ndarray] = None
        self.original_audio_data: Optional[np.ndarray] = None
        self.playback_sr: int = 16000
        self.playback_pos: float = 0.0
        self.stop_playback_event = threading.Event()

        # ================================================================
        # САЙДБАР (прокручиваемый)
        # ================================================================
        self.sidebar = ctk.CTkScrollableFrame(self, width=300, corner_radius=0,
                                              fg_color="#1a1a1a",
                                              scrollbar_button_color="#2b2b2b")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        # Логотип
        lbl_logo = ctk.CTkLabel(self.sidebar, text="🎙 QazTriber",
                                font=ctk.CTkFont(size=24, weight="bold"),
                                text_color="#ffffff")
        lbl_logo.grid(row=0, column=0, padx=20, pady=(25, 5))
        
        # Ссылка на автора в сайдбаре (дополнительно к меню)
        lbl_author = ctk.CTkLabel(self.sidebar, text="Связь с разработчиком",
                                  font=ctk.CTkFont(size=12, underline=True),
                                  text_color="#00aaff", cursor="hand2")
        lbl_author.grid(row=1, column=0, pady=(0, 20))
        lbl_author.bind("<Button-1>", lambda e: webbrowser.open("https://t.me/nurlan_nursultan"))

        # --- ВЫБОР ФАЙЛА ---
        self.btn_file = ctk.CTkButton(
            self.sidebar, text="🎵 Выбрать аудиофайл", height=40,
            fg_color="#2b2b2b", border_width=1, border_color="#444",
            hover_color="#363636", command=self.select_file)
        self.btn_file.grid(row=1, column=0, padx=16, pady=(5, 2), sticky="ew")
        self.lbl_file_hint = ctk.CTkLabel(
            self.sidebar, text="MP3 / WAV / M4A / FLAC / OGG",
            font=ctk.CTkFont(size=11), text_color="#666666")
        self.lbl_file_hint.grid(row=2, column=0, pady=(0, 12))

        # --- МОДЕЛЬ ---
        box_model = ctk.CTkFrame(self.sidebar, fg_color="#222222",
                                  corner_radius=10, border_width=1, border_color="#333")
        box_model.grid(row=3, column=0, padx=10, pady=(0, 8), sticky="ew")
        box_model.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(box_model, text="Модель Whisper",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#cccccc").pack(anchor="w", padx=12, pady=(10, 6))

        self.load_method_var = ctk.StringVar(value="auto")
        self.rb_auto = ctk.CTkRadioButton(box_model, text="Авто-скачивание (онлайн)",
                                           variable=self.load_method_var, value="auto",
                                           command=self.toggle_model_ui, text_color="#cccccc")
        self.rb_auto.pack(anchor="w", padx=12, pady=2)
        self.rb_local = ctk.CTkRadioButton(box_model, text="Локальный файл (.pt)",
                                            variable=self.load_method_var, value="local",
                                            command=self.toggle_model_ui, text_color="#cccccc")
        self.rb_local.pack(anchor="w", padx=12, pady=2)
        self.rb_saved = ctk.CTkRadioButton(box_model, text="Скачанные (offline)",
                                            variable=self.load_method_var, value="saved",
                                            command=self.toggle_model_ui, text_color="#cccccc")
        self.rb_saved.pack(anchor="w", padx=12, pady=(2, 8))

        # Авто-режим
        self.frame_auto = ctk.CTkFrame(box_model, fg_color="transparent")
        self.frame_auto.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(self.frame_auto, text="Размер модели:",
                     font=ctk.CTkFont(size=11), text_color="#888888").pack(anchor="w")
        self.quality_map = {
            "🚀 Medium (баланс)": "medium",
            "⚡ Tiny (быстро)": "tiny",
            "💨 Base": "base",
            "⚖️ Small": "small",
            "💎 Large (точность)": "large",
            "🔥 Turbo": "turbo",
            "✨ Свой (Hugging Face)": "custom"
        }
        self.quality_var = ctk.StringVar(value="🚀 Medium (баланс)")
        self.quality_menu = ctk.CTkOptionMenu(
            self.frame_auto, values=list(self.quality_map.keys()),
            variable=self.quality_var, fg_color="#2b2b2b",
            command=self.toggle_custom_model_ui)
        self.quality_menu.pack(fill="x", pady=(4, 0))
        self.frame_custom_hf = ctk.CTkFrame(self.frame_auto, fg_color="transparent")
        ctk.CTkLabel(self.frame_custom_hf, text="HuggingFace repo ID:",
                     font=ctk.CTkFont(size=11), text_color="#888888").pack(anchor="w")
        self.entry_custom_hf = ctk.CTkEntry(self.frame_custom_hf,
                                             placeholder_text="abilmansplus/whisper-turbo-ksc2")
        self.entry_custom_hf.pack(fill="x")
        self._add_context_menu(self.entry_custom_hf)

        # Локальный файл
        self.frame_local = ctk.CTkFrame(box_model, fg_color="transparent")
        self.btn_select_model = ctk.CTkButton(
            self.frame_local, text="📁 Выбрать файл модели",
            fg_color="#333", hover_color="#444", command=self.select_model)
        self.btn_select_model.pack(fill="x", padx=10, pady=5)
        self.lbl_model_hint = ctk.CTkLabel(
            self.frame_local, text="Файл не выбран",
            font=ctk.CTkFont(size=11), text_color="#666666")
        self.lbl_model_hint.pack()

        # Скачанные
        self.frame_saved = ctk.CTkFrame(box_model, fg_color="transparent")
        self.saved_models_var = ctk.StringVar(value="Нет моделей")
        self.saved_models_menu = ctk.CTkOptionMenu(
            self.frame_saved, values=["Нет моделей"],
            variable=self.saved_models_var, fg_color="#2b2b2b")
        self.saved_models_menu.pack(fill="x", padx=10, pady=(5, 2))
        ctk.CTkButton(self.frame_saved, text="🗑 Удалить выбранную", height=28,
                       fg_color="#3a3a3a", hover_color="#8b0000",
                       command=self.delete_selected_model).pack(fill="x", padx=10, pady=(0, 6))

        # Скрываем доп. фреймы
        self.frame_local.pack_forget()
        self.frame_saved.pack_forget()

        # --- ПОДСКАЗКА ---
        box_prompt = ctk.CTkFrame(self.sidebar, fg_color="#222222",
                                   corner_radius=10, border_width=1, border_color="#333")
        box_prompt.grid(row=4, column=0, padx=10, pady=(0, 8), sticky="ew")
        ctk.CTkLabel(box_prompt, text="Языковая подсказка (prompt)",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#cccccc").pack(anchor="w", padx=12, pady=(10, 2))
        ctk.CTkLabel(box_prompt,
                     text="Помогает распознавать смесь каз./рус.\nОставьте или измените пример.",
                     font=ctk.CTkFont(size=10), text_color="#666666", justify="left").pack(anchor="w", padx=12)
        self.entry_prompt = ctk.CTkEntry(box_prompt, height=34)
        self.entry_prompt.insert(0, DEFAULT_PROMPT)
        self.entry_prompt.pack(fill="x", padx=10, pady=(5, 10))
        self._add_context_menu(self.entry_prompt)

        # --- ЗАПИСЬ С МИКРОФОНА ---
        box_rec = ctk.CTkFrame(self.sidebar, fg_color="#222222",
                                corner_radius=10, border_width=1, border_color="#333")
        box_rec.grid(row=5, column=0, padx=10, pady=(0, 8), sticky="ew")
        ctk.CTkLabel(box_rec, text="Микрофон",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#cccccc").pack(anchor="w", padx=12, pady=(10, 4))
        self.btn_record = ctk.CTkButton(
            box_rec, text="🎤 Начать запись", height=38,
            fg_color="#7a1a00", hover_color="#aa2200",
            command=self.toggle_recording)
        self.btn_record.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkButton(box_rec, text="📂 Открыть папку записей", height=30,
                       fg_color="#2b2b2b", hover_color="#363636",
                       command=self.open_recordings_folder).pack(fill="x", padx=10, pady=(0, 10))

        # --- КНОПКА ЗАПУСКА ---
        self.btn_run = ctk.CTkButton(
            self.sidebar, text="🚀 Начать транскрибацию",
            height=52, font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#0055bb", hover_color="#0044aa",
            text_color="#ffffff", command=self.start_process)
        self.btn_run.grid(row=6, column=0, padx=16, pady=(4, 30), sticky="ew")

        # ================================================================
        # ГЛАВНАЯ ПАНЕЛЬ
        # ================================================================
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=28, pady=28)
        self.main_view.grid_columnconfigure(0, weight=1)
        # row 0 = статус, row 1 = плеер (скрытый), row 2 = текст, row 3 = кнопки
        self.main_view.grid_rowconfigure(2, weight=1)

        # --- СТАТУС / ПРОГРЕСС (row 0) ---
        self.info_frame = ctk.CTkFrame(self.main_view, fg_color="#1e1e1e", corner_radius=12,
                                        border_width=1, border_color="#2a2a2a")
        self.info_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.info_frame.grid_columnconfigure(0, weight=1)

        row_status = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        row_status.pack(fill="x", padx=15, pady=(10, 4))
        row_status.grid_columnconfigure(0, weight=1)
        self.lbl_info = ctk.CTkLabel(
            row_status, text="🟢 Готов к работе. Выберите аудиофайл.",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#00cc66", anchor="w")
        self.lbl_info.grid(row=0, column=0, sticky="w")
        self.lbl_percent = ctk.CTkLabel(
            row_status, text="", font=ctk.CTkFont(size=12),
            text_color="#00cc66", width=40, anchor="e")
        self.lbl_percent.grid(row=0, column=1, sticky="e", padx=(8, 0))
        # Кнопка принудительной остановки
        self.btn_stop = ctk.CTkButton(
            row_status, text="⏹ Стоп", width=70, height=26,
            fg_color="#5a1010", hover_color="#8b0000",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.force_stop_asr, state="disabled")
        self.btn_stop.grid(row=0, column=2, sticky="e", padx=(8, 0))

        self.p_bar = ctk.CTkProgressBar(self.info_frame, height=8,
                                         fg_color="#2a2a2a", progress_color="#00cc66")
        self.p_bar.pack(fill="x", padx=15, pady=(0, 12))
        self.p_bar.set(0)

        # --- ПЛЕЕР (row 1, скрыт по умолчанию) ---
        self.frame_player = ctk.CTkFrame(self.main_view, fg_color="#1e1e1e", corner_radius=12,
                                          border_width=1, border_color="#2a2a2a")
        
        # Интеграция визуализатора волн в правую панель
        self.waveform = WaveformWidget(self.frame_player, on_crop=self._apply_crop, on_seek=self.seek_playback)
        self.waveform.pack(fill="x", padx=14, pady=(14, 0))

        player_row = ctk.CTkFrame(self.frame_player, fg_color="transparent")
        player_row.pack(fill="x", padx=14, pady=(8, 14))
        self.btn_play_pause = ctk.CTkButton(
            player_row, text="▶", width=40, height=40,
            fg_color="#1a4a99", hover_color="#2255bb",
            font=ctk.CTkFont(size=17), command=self.toggle_playback)
        self.btn_play_pause.pack(side="left", padx=(0, 10))

        self.btn_undo = ctk.CTkButton(
            player_row, text="↩️ Вернуть", height=30, width=80,
            fg_color="#333333", hover_color="#555555",
            font=ctk.CTkFont(size=12), command=self.undo_audio)
        self.btn_undo.pack(side="left", padx=(10, 0))

        self.btn_amplify = ctk.CTkButton(
            player_row, text="⚡ Нормализовать", height=30, width=120,
            fg_color="#333333", hover_color="#555555",
            font=ctk.CTkFont(size=12), command=self.amplify_audio)
        self.btn_amplify.pack(side="left", padx=(10, 0))

        self.lbl_duration = ctk.CTkLabel(
            player_row, text="0:00 / 0:00",
            font=ctk.CTkFont(size=11), text_color="#aaaaaa")
        self.lbl_duration.pack(side="right", padx=(10, 0))

        # --- ТЕКСТБОКС (row 2) ---
        self.output_box = ctk.CTkTextbox(
            self.main_view, corner_radius=12, fg_color="#0f0f0f",
            border_width=1, border_color="#2a2a2a",
            font=("Georgia", 15), text_color="#e0e0e0",
            padx=20, pady=16)
        self.output_box.grid(row=2, column=0, sticky="nsew")
        self.output_box.insert("1.0",
            "Здесь появится распознанный текст.\n\n"
            "Инструкция:\n"
            "1. Выберите аудиофайл или запишите с микрофона.\n"
            "2. Выберите модель.\n"
            "3. Нажмите «Начать транскрибацию».")
        self._add_context_menu(self.output_box)

        # --- КНОПКИ СНИЗУ (row 3) ---
        bottom = ctk.CTkFrame(self.main_view, fg_color="transparent")
        bottom.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        bottom.grid_columnconfigure(0, weight=1)
        self.btn_save = ctk.CTkButton(
            bottom, text="💾 Сохранить как TXT", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2b2b2b", hover_color="#3a3a3a",
            command=self.save_text)
        self.btn_save.pack(side="right", padx=(6, 0))
        self.btn_copy_all = ctk.CTkButton(
            bottom, text="📋 Скопировать всё", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2b2b2b", hover_color="#3a3a3a",
            command=self.copy_all_text)
        self.btn_copy_all.pack(side="right")
        
        self.btn_reset = ctk.CTkButton(
            bottom, text="🗑 Сброс", height=40, width=100,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#8b0000", hover_color="#aa0000",
            command=self.reset_app)
        self.btn_reset.pack(side="left", padx=(10, 0))

        # Горячие клавиши
        if sys.platform == "darwin":
            self.bind_all("<KeyPress>", self._handle_hotkey)

        # Загрузка списка моделей
        self.after(300, self.refresh_model_menu)

    # ================================================================
    # КОНТЕКСТНОЕ МЕНЮ (правая кнопка / горячие клавиши)
    # ================================================================
    def _add_context_menu(self, widget):
        menu = Menu(self, tearoff=0, bg="#2b2b2b", fg="#ffffff",
                    activebackground="#0055bb", font=("Helvetica", 12))
        menu.add_command(label="Копировать   ⌘C", command=lambda: self._copy(None, widget))
        menu.add_command(label="Вставить      ⌘V", command=lambda: self._paste(None, widget))
        menu.add_command(label="Вырезать     ⌘X", command=lambda: self._cut(None, widget))
        menu.add_separator()
        menu.add_command(label="Выделить всё ⌘A", command=lambda: self._select_all(None, widget))
        btn = "<Button-2>" if sys.platform == "darwin" else "<Button-3>"
        widget.bind(btn, lambda e: menu.tk_popup(e.x_root, e.y_root))
        # Также привязываем Cmd+клавиши прямо к виджету
        widget.bind("<Command-c>", lambda e: self._copy(e, widget))
        widget.bind("<Command-v>", lambda e: self._paste(e, widget))
        widget.bind("<Command-x>", lambda e: self._cut(e, widget))
        widget.bind("<Command-a>", lambda e: self._select_all(e, widget))

    def _paste(self, event, widget=None):
        if widget is None: widget = self.focus_get()
        if widget and hasattr(widget, "insert"):
            try:
                content = self.clipboard_get()
                if isinstance(widget, ctk.CTkTextbox):
                    try: widget.delete("sel.first", "sel.last")
                    except: pass
                    widget.insert("insert", content)
                else:
                    try: widget.delete("sel.first", "sel.last")
                    except: pass
                    widget.insert("insert", content)
                return "break"
            except: pass

    def _copy(self, event, widget=None):
        if widget is None: widget = self.focus_get()
        if widget:
            try:
                content = ""
                if isinstance(widget, ctk.CTkTextbox):
                    content = widget.get("sel.first", "sel.last")
                elif hasattr(widget, "selection_get"):
                    content = widget.selection_get()
                else:
                    content = widget.get()
                if content:
                    self.clipboard_clear()
                    self.clipboard_append(content)
                return "break"
            except: pass

    def _cut(self, event, widget=None):
        self._copy(event, widget)
        if widget is None: widget = self.focus_get()
        if widget and hasattr(widget, "delete"):
            try: widget.delete("sel.first", "sel.last"); return "break"
            except: pass

    def _select_all(self, event, widget=None):
        if widget is None: widget = self.focus_get()
        if widget:
            if isinstance(widget, ctk.CTkEntry):
                widget.select_range(0, "end"); widget.icursor("end")
            elif isinstance(widget, ctk.CTkTextbox):
                widget.tag_add("sel", "1.0", "end")
            return "break"

    def _handle_hotkey(self, event):
        """Глобальный перехват горячих клавиш macOS."""
        cmd = (event.state & 0x8) or (event.state & 0x10) or (event.state & 0x100)
        if not cmd: return
        key = event.keysym.lower()
        if key == "v": return self._paste(event)
        elif key == "c": return self._copy(event)
        elif key == "a": return self._select_all(event)
        elif key == "x": return self._cut(event)

    # ================================================================
    # УПРАВЛЕНИЕ МОДЕЛЯМИ
    # ================================================================
    def refresh_model_menu(self):
        new_values = []
        for display, internal in self.quality_map.items():
            if internal == "custom":
                new_values.append(display); continue
            suffix = " ✓" if WhisperEngine.is_model_downloaded(internal, MODELS_DIR) else ""
            new_values.append(f"{display}{suffix}")
        self.quality_menu.configure(values=new_values)

    def toggle_model_ui(self):
        method = self.load_method_var.get()
        if method == "auto":
            self.frame_local.pack_forget(); self.frame_saved.pack_forget()
            self.frame_auto.pack(fill="x", padx=10, pady=(0, 8))
            self.toggle_custom_model_ui()
        elif method == "local":
            self.frame_auto.pack_forget(); self.frame_saved.pack_forget()
            self.frame_local.pack(fill="x", padx=10, pady=(0, 8))
        else:
            self.frame_auto.pack_forget(); self.frame_local.pack_forget()
            self.frame_saved.pack(fill="x", padx=10, pady=(0, 8))
            self.refresh_saved_models_list()

    def toggle_custom_model_ui(self, choice=None):
        q = self.quality_var.get().replace(" ✓", "")
        if q == "✨ Свой (Hugging Face)":
            self.frame_custom_hf.pack(fill="x", pady=(5, 0))
        else:
            self.frame_custom_hf.pack_forget()

    def refresh_saved_models_list(self):
        models = WhisperEngine.get_downloaded_models(MODELS_DIR)
        if not models:
            self.saved_models_menu.configure(values=["Кэш пуст"])
            self.saved_models_var.set("Кэш пуст")
        else:
            names = [m["name"] for m in models]
            self.saved_models_menu.configure(values=names)
            if self.saved_models_var.get() not in names:
                self.saved_models_var.set(names[0])

    def delete_selected_model(self):
        name = self.saved_models_var.get()
        if name in ["Кэш пуст", "Нет моделей"]: return
        if messagebox.askyesno("Удаление", f"Удалить '{name}'?"):
            try:
                WhisperEngine.delete_model(name, MODELS_DIR)
                messagebox.showinfo("Готово", f"Модель '{name}' удалена.")
                self.refresh_saved_models_list(); self.refresh_model_menu()
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def select_model(self):
        f = filedialog.askopenfilename(filetypes=[("Whisper", "*.pt *.bin")])
        if f:
            self.model_path = f
            self.lbl_model_hint.configure(text=os.path.basename(f), text_color="#00cc66")

    # ================================================================
    # ВЫБОР ФАЙЛА И ПЛЕЕР
    # ================================================================
    def select_file(self):
        f = filedialog.askopenfilename(
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.ogg *.flac")])
        if f:
            self.audio_path = f
            self.lbl_file_hint.configure(text=f"⏳ Загрузка...", text_color="#aaaaaa")
            self._update_status("⏳ Обработка длинного файла: это можёт занять несколько секунд...")
            self.update_idletasks() # Принудительно обновить UI

            def _load():
                try:
                    y, sr = librosa.load(f, sr=16000, mono=True)
                    self.playback_data = y.astype(np.float32)
                    self.original_audio_data = self.playback_data.copy()
                    self.playback_sr = 16000
                    self.playback_pos = 0.0
                    self.after(0, self._on_audio_loaded, f)
                except Exception as e:
                    print(f"Плеер: {e}")
                    self.after(0, self._update_status, f"❌ Ошибка: {e}")
            threading.Thread(target=_load, daemon=True).start()

    def _on_audio_loaded(self, filepath):
        self.lbl_file_hint.configure(text=os.path.basename(filepath), text_color="#00cc66")
        self._update_status("✅ Файл загружен в память. Готов к работе.")
        self.show_player_ui()

    def show_player_ui(self):
        if self.playback_data is None: return
        self.playback_pos = 0.0
        self.waveform.show_waveform(self.playback_data, self.playback_sr)
        self.btn_play_pause.configure(text="▶")
        self._refresh_player_label()
        self.frame_player.grid(row=1, column=0, sticky="ew", pady=(0, 8))

    def _refresh_player_label(self):
        if self.playback_data is None: return
        curr = self.playback_pos / self.playback_sr
        total = float(len(self.playback_data)) / self.playback_sr
        self.lbl_duration.configure(
            text=f"{int(curr//60)}:{int(curr%60):02d} / {int(total//60)}:{int(total%60):02d}")
        frac = curr / max(total, 1e-5)
        self.waveform.set_progress(frac)

    def toggle_playback(self):
        if self.is_playing: self.stop_playback()
        else: self.start_playback()

    def start_playback(self):
        if self.playback_data is None: return
        self.is_playing = True
        self.btn_play_pause.configure(text="⏸")
        self.stop_playback_event.clear()

        def play_thread():
            try:
                # Большой blocksize убирает треск
                BLOCK = 4096

                def cb(outdata, frames, t, status):
                    if self.stop_playback_event.is_set():
                        outdata[:] = 0; raise sd.CallbackStop
                    s = int(self.playback_pos)
                    e = s + frames
                    if self.playback_data is None or s >= len(self.playback_data):
                        outdata[:] = 0; self.playback_pos = 0.0
                        self.is_playing = False
                        self.after(0, self._on_playback_end); raise sd.CallbackStop
                    if e > len(self.playback_data):
                        avail = len(self.playback_data) - s
                        outdata[:avail, 0] = self.playback_data[s:s+avail]
                        outdata[avail:, 0] = 0.0
                        self.playback_pos = 0.0; self.is_playing = False
                        self.after(0, self._on_playback_end); raise sd.CallbackStop
                    outdata[:, 0] = self.playback_data[s:e]
                    self.playback_pos = float(e)
                    # Обновляем UI ~4 раза в секунду
                    if int(self.playback_pos) % max(1, self.playback_sr // 4) < frames:
                        self.after(0, self._refresh_player_label)

                with sd.OutputStream(samplerate=self.playback_sr, channels=1,
                                     dtype='float32', blocksize=BLOCK, callback=cb,
                                     latency='high'):
                    while self.is_playing and not self.stop_playback_event.is_set():
                        sd.sleep(50)
            except Exception as ex:
                print(f"Playback: {ex}")
                self.after(0, self.stop_playback)

        threading.Thread(target=play_thread, daemon=True).start()

    def _on_playback_end(self):
        self.is_playing = False
        self.btn_play_pause.configure(text="▶")
        self._refresh_player_label()

    def stop_playback(self):
        self.is_playing = False
        self.stop_playback_event.set()
        self.btn_play_pause.configure(text="▶")

    def seek_playback(self, frac: float):
        if self.playback_data is None: return
        self.playback_pos = max(0.0, min(float(frac) * len(self.playback_data),
                                          float(len(self.playback_data))))
        self._refresh_player_label()
        if self.is_playing:
            self.stop_playback()
            self.after(80, self.start_playback)

    def undo_audio(self):
        if not hasattr(self, 'original_audio_data') or self.original_audio_data is None: return
        self.playback_data = self.original_audio_data.copy()
        self.waveform.show_waveform(self.playback_data, self.playback_sr)
        self.playback_pos = 0.0
        self._refresh_player_label()
        self._update_status("↩️ Исходное аудио восстановлено.")

    def amplify_audio(self):
        if self.playback_data is None or len(self.playback_data) == 0: return
        peak = float(np.max(np.abs(self.playback_data)))
        if peak < 1e-4: return
        coeff = 1.0 / peak
        self.playback_data = self.playback_data * coeff
        self.waveform.show_waveform(self.playback_data, self.playback_sr)
        self._update_status(f"⚡ Звук нормализован (усиление x{coeff:.1f}).")

    # ================================================================
    # ЗАПИСЬ С МИКРОФОНА
    # ================================================================
    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_data = []
            self.btn_record.configure(text="🛑 Остановить запись", fg_color="#cc0000")
            self._update_status("🎤 Идёт запись...")
            
            # Показываем Waveform в правой панели во время записи
            self.waveform.clear()
            self.frame_player.grid(row=1, column=0, sticky="ew", pady=(0, 8))

            def cb(indata: np.ndarray, frames: int, t, status):
                if status: print(f"Rec status: {status}")
                chunk = indata[:, 0].copy()
                self.recording_data.append(chunk)
                # Визуализация VU-метра в реальном времени
                self.after(0, lambda c=chunk: self.waveform.push_live(c))

            try:
                self.stream = sd.InputStream(
                    samplerate=16000, channels=1, dtype='float32',
                    blocksize=1024, callback=cb, latency='low')
                self.stream.start()
            except Exception as e:
                self.is_recording = False
                self.btn_record.configure(text="🎤 Начать запись", fg_color="#7a1a00")
                messagebox.showerror("Микрофон", f"Ошибка:\n{e}")
        else:
            self.is_recording = False
            self.btn_record.configure(text="🎤 Начать запись", fg_color="#7a1a00")
            try:
                if self.stream:
                    self.stream.stop(); self.stream.close(); self.stream = None
            except: pass

            if not self.recording_data:
                self._update_status("❌ Пустая запись — нет данных с микрофона")
                return

            audio_np = np.concatenate(self.recording_data, axis=0).astype(np.float32)
            rms = float(np.sqrt(np.mean(audio_np ** 2)))
            if rms < 1e-5:
                messagebox.showwarning("Тишина",
                    "Запись содержит только тишину.\n\n"
                    "Проверьте:\n• Разрешение микрофона в Настройки → Безопасность")
                self._update_status("⚠️ Запись — тишина. Проверьте микрофон.")
                return

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"rec_{ts}.wav"
            fpath = os.path.join(RECORDINGS_DIR, fname)
            sf.write(fpath, audio_np, 16000)

            self.audio_path = fpath
            self.playback_data = audio_np
            self.original_audio_data = self.playback_data.copy()
            self.playback_sr = 16000
            self.playback_pos = 0.0
            self.lbl_file_hint.configure(text=f"🎤 {fname}", text_color="#00cc66")
            self._update_status(f"✅ Запись сохранена: {fname}")
            # Показываем полную волновую форму для выделения региона
            self.after(0, lambda: self.waveform.show_waveform(audio_np, 16000))
            self.after(0, self.show_player_ui)

    def open_recordings_folder(self):
        import subprocess
        if sys.platform == "darwin":
            subprocess.run(["open", RECORDINGS_DIR])
        else:
            try: subprocess.run(["xdg-open", RECORDINGS_DIR])
            except: pass

    # ================================================================
    # СОХРАНЕНИЕ
    # ================================================================
    def save_text(self):
        content = self.output_box.get("1.0", "end-1c").strip()
        if not content or content.startswith("Здесь появится"):
            messagebox.showwarning("Внимание", "Нет текста для сохранения!"); return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Текст", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            messagebox.showinfo("Сохранено", "Файл сохранён!")

    def copy_all_text(self):
        """Copies all text from the output box to clipboard."""
        content = self.output_box.get("1.0", "end-1c").strip()
        if not content or content.startswith("Здесь появится"):
            messagebox.showwarning("Внимание", "Нечего копировать!"); return
        self.clipboard_clear()
        self.clipboard_append(content)
        # Визуальная обратная связь
        self.btn_copy_all.configure(text="✅ Скопировано!")
        self.after(2000, lambda: self.btn_copy_all.configure(text="📋 Скопировать всё"))

    def reset_app(self):
        """Полный сброс приложения в исходное состояние."""
        # 1. Останавливаем воспроизведение
        if self.is_playing:
            self.stop_playback()
        
        # 2. Очищаем данные аудио
        self.audio_path = None
        self.playback_data = None
        self.original_audio_data = None
        self.recording_data = []
        self.playback_pos = 0.0
        
        # 3. Прячем интерфейс плеера и визуализатора
        self.waveform.clear()
        self.frame_player.grid_forget()
        
        # 4. Сбрасываем лейбл файла
        self.lbl_file_hint.configure(text="Файл не выбран", text_color="#666666")
        
        # 5. Очищаем окно вывода текста
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0",
            "Здесь появится распознанный текст.\n\n"
            "Инструкция:\n"
            "1. Выберите аудиофайл или запишите с микрофона.\n"
            "2. Выберите модель.\n"
            "3. Нажмите «Начать транскрибацию».")
            
        # 6. Обновляем статус
        self.p_bar.set(0)
        self.lbl_percent.configure(text="")
        self._update_status("🟢 Готов к работе. Выберите аудиофайл.", color="#00cc66")

    def force_stop_asr(self):
        """Немедленно устанавливает stop_event — прерывает между чанками."""
        self.stop_event.set()
        self.stop_pulsing = True
        self._update_status("⏹ Остановка после текущего чанка...")
        self.btn_stop.configure(state="disabled", text="⏳ Стоп...")

    def _apply_crop(self, audio: np.ndarray, sr: int):
        """Callback от WaveformWidget: сохраняет обрезанный кусок как новый audio_path."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"crop_{ts}.wav"
        fpath = os.path.join(RECORDINGS_DIR, fname)
        sf.write(fpath, audio, sr)
        self.audio_path = fpath
        self.playback_data = audio
        self.playback_sr = sr
        self.playback_pos = 0.0
        self.lbl_file_hint.configure(text=f"✂️ {fname}", text_color="#ffaa00")
        self._update_status(f"✂️ Обрезок: {len(audio)/sr:.1f} сек. Готов к транскрибации.")
        # Отобразить обрезанный участок на всю ширину волны
        self.waveform.show_waveform(audio, sr)
        self.after(0, self.show_player_ui)

    # ================================================================
    # СТАТУС И ПРОГРЕСС
    # ================================================================
    def _update_status(self, text: str, progress: Optional[float] = None):
        def _apply():
            self.lbl_info.configure(text=text)
            if progress is not None:
                self.p_bar.stop()
                self.p_bar.configure(mode="determinate")
                self.p_bar.set(progress)
                pct = int(progress * 100)
                self.lbl_percent.configure(text=f"{pct}%")
        self.after(0, _apply)

    def _toggle_pulsing(self, on: bool, base_text: Optional[str] = None):
        if not on:
            self.stop_pulsing = True
            return
        self.stop_pulsing = False
        def pulse():
            dots = 0
            while not self.stop_pulsing:
                dots = (dots + 1) % 4
                t = (base_text or "Обработка") + "." * dots
                self.after(0, lambda t=t: self.lbl_info.configure(text=t))
                time.sleep(0.5)
        threading.Thread(target=pulse, daemon=True).start()

    # ================================================================
    # ТРАНСКРИБАЦИЯ
    # ================================================================
    def start_process(self):
        if not self.audio_path:
            messagebox.showwarning("Файл", "Выберите аудиофайл!"); return
        method = self.load_method_var.get()
        if method == "local" and not self.model_path:
            messagebox.showwarning("Модель", "Укажите файл модели!"); return
        self.btn_run.configure(state="disabled")
        self.stop_event.clear()  # сбрасываем Event
        self.btn_stop.configure(state="normal", text="⏹ Стоп")
        self.output_box.delete("1.0", "end")
        self.lbl_percent.configure(text="")
        self.p_bar.set(0)
        threading.Thread(target=self.run_asr_logic, daemon=True).start()

    def run_asr_logic(self):
        try:
            method = self.load_method_var.get()
            if method == "auto":
                q = self.quality_var.get().replace(" ✓", "")
                variant = self.quality_map.get(q, "medium")
                if variant == "custom":
                    target = self.entry_custom_hf.get().strip()
                    if not target: raise ValueError("Введите HuggingFace repo ID!")
                else:
                    target = variant
            elif method == "saved":
                target = self.saved_models_var.get()
                if target in ["Кэш пуст", "Нет моделей"]:
                    raise ValueError("Выберите модель из списка!")
            else:
                target = self.model_path
                if not target: raise ValueError("Укажите путь к файлу модели!")

            # 1. ШАГ: Загрузка модели (MB-прогресс внутри load_model)
            self._update_status("🧠 Загрузка модели...", 0.0)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.engine = WhisperEngine()
            self.engine.load_model(
                model_path=target, device=device,
                beam_size=5, callback=self._update_status,
                cache_dir=MODELS_DIR
            )
            self.after(0, self.refresh_model_menu)

            if self.stop_event.is_set(): raise InterruptedError("Остановлено")

            # 2. ШАГ: Чтение аудио
            self._update_status("🎵 Чтение аудиофайла...", 0.0)
            y, sr = librosa.load(self.audio_path, sr=16000, mono=True)
            speech = y.astype(np.float32)
            self._update_status(f"✅ Аудио: {len(speech)/16000:.1f} сек", 1.0)

            if self.stop_event.is_set(): raise InterruptedError("Остановлено")

            # 3. ШАГ: Транскрибация (по чанкам, прогресс 0→100%)
            self._update_status("⚙️ Транскрибация...", 0.0)
            prompt = self.entry_prompt.get().strip() or None
            try:
                text = self.engine.transcribe(
                    speech, sr=16000,
                    initial_prompt=prompt,
                    progress_callback=self._update_status,
                    stop_event=self.stop_event
                )
                if not text or text.strip() in ["!", "!!", ".", "...", " ", ""]:
                    text = "[Не удалось распознать речь — тишина или шум]"
            finally:
                self._toggle_pulsing(False)

            # 4. Вывод
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            def _show():
                self.output_box.insert("end", f"[{ts}]\n{text}\n\n")
                self.output_box.see("end")
            self.after(0, _show)
            self._update_status("✅ Готово!", 1.0)

            # 5. Очистка памяти
            if self.engine:
                self.engine.unload_model()
                self.engine = None

            self.after(0, lambda: messagebox.showinfo("Готово", "Транскрибация завершена!\nПамять освобождена."))

        except InterruptedError:
            # Пользователь нажал Стоп
            self._update_status("⏹ Остановлено пользователем")
            if self.engine:
                try: self.engine.unload_model()
                except: pass
                self.engine = None
        except Exception as e:
            self._update_status("❌ Ошибка!")
            err = str(e)
            def _err():
                self.output_box.insert("end",
                    f"\n[ОШИБКА]: {err}\n\n"
                    "Проверьте:\n"
                    "• ffmpeg: brew install ffmpeg\n"
                    "• Доступность модели\n"
                    "• Достаточно ли памяти\n")
                self.output_box.see("end")
            self.after(0, _err)
        finally:
            self.p_bar.stop()
            self.p_bar.configure(mode="determinate")
            self.p_bar.set(1)
            self.btn_run.configure(state="normal")
            self.btn_stop.configure(state="disabled", text="⏹ Стоп")


# --- НАСТРОЙКА ПУТЕЙ ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "../../.."))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)


if __name__ == "__main__":
    app = QazTriberApp()
    app.mainloop()