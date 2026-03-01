import os
import sys
import threading
import gc
import socket
import webbrowser
import numpy as np

# --- RESOURCE PATH FIX ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        # For PyInstaller 6+, check for _internal folder
        internal_path = os.path.join(base_path, "_internal")
        if os.path.exists(internal_path):
            base_path = internal_path
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- DEBUG BOOT ---
print("🚀 QazTriber Booting...")
print(f"Current Dir: {os.getcwd()}")
print(f"Python: {sys.version}")
if getattr(sys, 'frozen', False):
    print(f"Frozen mode! MEIPASS: {getattr(sys, '_MEIPASS', 'not set')}")
# ------------------

import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox, Menu
from typing import Optional, Callable
from datetime import datetime

# Import simplified engine
from engine.whisper_engine import WhisperEngine

# ================================================================
# SINGLE-INSTANCE PROTECTION
# ================================================================
_LOCK_PORT = 47182
_lock_socket = None # Keep reference

def _is_already_running() -> bool:
    global _lock_socket
    try:
        _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _lock_socket.bind(("127.0.0.1", _LOCK_PORT))
        _lock_socket.listen(1)
        return False
    except OSError:
        return True

if _is_already_running():
    print("⚠️ Instance already running. Exiting.")
    sys.exit(0)

# ================================================================
# STYLING CONSTANTS (Premium macOS Dark)
# ================================================================
BG_APP = "#0A0A0B"
BG_SURFACE = "#161618"
ACCENT_BLUE = "#007AFF"
ACCENT_RED = "#FF3B30"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#A1A1A6"
BORDER_COL = "#2C2C2E"

ctk.set_appearance_mode("dark")

# ================================================================
# WAVEFORM WIDGET (Polished)
# ================================================================
class WaveformWidget:
    CANVAS_H = 120
    WAVE_COL = ACCENT_BLUE
    SEL_COL  = "#1C1C1E"
    PROG_COL = "#FFFFFF"

    def __init__(self, parent, on_seek, on_crop):
        self._on_seek = on_seek
        self._on_crop = on_crop
        self.audio_data = None
        self.display_data = None
        self.sr = 16000
        self.play_progress = 0.0
        self.sel_start = None
        self.sel_end = None
        self._drag_x0 = None

        self.frame = ctk.CTkFrame(parent, fg_color=BG_SURFACE, corner_radius=12, border_width=1, border_color=BORDER_COL)
        self.canvas = tk.Canvas(self.frame, bg=BG_SURFACE, height=self.CANVAS_H, highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="x", padx=10, pady=(10, 2))

        # Scissors / Crop button (hidden by default)
        self.btn_crop = ctk.CTkButton(self.frame, text="✂️ Обрезать выделенное", height=28, 
                                       fg_color="#333", hover_color="#555", font=ctk.CTkFont(size=11), 
                                       command=self._do_crop)
        
        self.live_rms = []
        self._MAX_LIVE = 150

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

    def show_audio(self, audio: np.ndarray, sr: int = 16000):
        self.audio_data = audio
        self.sr = sr
        self.play_progress = 0.0
        self.sel_start = self.sel_end = None
        self.btn_crop.pack_forget() # Hide on new audio
        
        # Optimized downsampling for 2-3 hour files
        target_pts = 1500
        n = len(audio)
        if n < 1: return
        
        step = max(1, n // target_pts)
        try:
            if n > 5000000:
                reduced = []
                chunk_size = 1000000
                for i in range(0, n, chunk_size):
                    chunk = np.abs(audio[i:i+chunk_size])
                    c_step = max(1, len(chunk) // (target_pts * chunk_size // n + 1))
                    if c_step > 0:
                        trimmed = chunk[:(len(chunk)//c_step)*c_step]
                        if len(trimmed) > 0:
                            reduced.extend(np.max(trimmed.reshape(-1, c_step), axis=1))
                self.display_data = np.array(reduced, dtype=np.float32)
            else:
                trimmed = np.abs(audio[:(n // step) * step])
                self.display_data = np.max(trimmed.reshape(-1, step), axis=1)
        except:
            self.display_data = np.abs(audio[::step])
            
        self._redraw()

    def push_live(self, rms: float):
        """Update live waveform during recording"""
        self.live_rms.append(rms)
        if len(self.live_rms) > self._MAX_LIVE:
            self.live_rms.pop(0)
        self._draw_live()

    def _draw_live(self):
        c = self.canvas
        w = c.winfo_width()
        h = self.CANVAS_H
        if w <= 1 or not self.live_rms: return
        
        c.delete("all")
        mid = h // 2
        bw = max(2, w // self._MAX_LIVE)
        
        # Draw background bars
        peak = max(max(self.live_rms), 1e-5)
        for i, val in enumerate(self.live_rms):
            norm = val / peak
            bh = int(norm * mid * 0.9)
            x = i * bw
            color = "#FF3B30" if norm > 0.7 else "#FF9500" if norm > 0.4 else ACCENT_BLUE
            c.create_rectangle(x, mid - bh, x + bw - 1, mid + bh, fill=color, outline="", tags="wave")

    def set_progress(self, frac: float):
        self.play_progress = max(0.0, min(1.0, frac))
        self._redraw()

    def _redraw(self):
        c = self.canvas
        w = c.winfo_width()
        h = self.CANVAS_H
        if w <= 1 or self.audio_data is None: return
        c.delete("all")

        mid = h // 2
        # Selection highlight
        if self.sel_start is not None and self.sel_end is not None:
            total_len = len(self.audio_data)
            s_smpl, e_smpl = sorted([self.sel_start, self.sel_end])
            x1 = (s_smpl / total_len) * w
            x2 = (e_smpl / total_len) * w
            c.create_rectangle(x1, 0, x2, h, fill="#2C2C2E", outline="")

        # Wave rendering
        if self.display_data is not None:
            pts = len(self.display_data)
            dx = w / pts
            for i, val in enumerate(self.display_data):
                bh = int(val * mid * 0.9)
                if bh > 0:
                    x = i * dx
                    c.create_line(x, mid - bh, x, mid + bh, fill=self.WAVE_COL, width=1)

        # Progress mark
        px = self.play_progress * w
        c.create_line(px, 0, px, h, fill=self.PROG_COL, width=2)

    def _on_press(self, e):
        if self.audio_data is None: return
        self._drag_x0 = e.x
        self.sel_start = (e.x / self.canvas.winfo_width()) * len(self.audio_data)
        self.sel_end = None

    def _on_drag(self, e):
        if self.audio_data is None: return
        self.sel_end = max(0, min(len(self.audio_data), (e.x / self.canvas.winfo_width()) * len(self.audio_data)))
        self._redraw()

    def _on_release(self, e):
        if self.audio_data is None: return
        if self._drag_x0 is not None and abs(e.x - self._drag_x0) < 5:
            # Click -> Seek
            self.sel_start = self.sel_end = None
            self.btn_crop.pack_forget()
            frac = max(0.0, min(1.0, e.x / self.canvas.winfo_width()))
            self._on_seek(frac)
        else:
            # Multi-select
            if self.sel_start is not None and self.sel_end is not None:
                self.btn_crop.pack(fill="x", padx=10, pady=(0, 10))
            self._redraw()
        self._drag_x0 = None

    def _do_crop(self):
        if self.audio_data is None or self.sel_start is None or self.sel_end is None: return
        s, e = sorted([int(self.sel_start), int(self.sel_end)])
        self._on_crop(self.audio_data[s:e], self.sr)
        self.sel_start = self.sel_end = None
        self.btn_crop.pack_forget()

# ================================================================
# MAIN APPLICATION
# ================================================================
class QazTriberApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("QazTriber Pro")
        self.geometry("1100x850")
        self.configure(fg_color=BG_APP)

        # --- STATE ---
        self.engine = WhisperEngine()
        self.audio_data = None
        self.sr = 16000
        self.is_recording = False
        self.is_playing = False
        self.playback_pos = 0.0
        self.stop_playback_event = threading.Event()
        self.recordings_dir = os.path.expanduser("~/Documents/QazTriber/recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)
        self._rec_buffer = []
        self.record_seconds = 0
        self._recording_timer_id = None
        self.stop_trans_event = threading.Event()
        self.is_transcribing = False

        self._setup_ui()

    def _setup_ui(self):
        self._setup_menu()
        # 1. Header Bar (Glassmorphism inspired)
        header = ctk.CTkFrame(self, fg_color=BG_SURFACE, height=70, corner_radius=0)
        header.pack(fill="x", side="top")
        
        lbl_title = ctk.CTkLabel(header, text="🎙 QazTriber Pro", font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRIMARY)
        lbl_title.pack(side="left", padx=30)

        self.lbl_status = ctk.CTkLabel(header, text="🟢 Готов к работе", font=ctk.CTkFont(size=14), text_color=ACCENT_BLUE)
        self.lbl_status.pack(side="left", padx=20)

        btn_help = ctk.CTkLabel(header, text="Помощь", font=ctk.CTkFont(size=13, underline=True), text_color=TEXT_SECONDARY, cursor="hand2")
        btn_help.pack(side="right", padx=30)
        btn_help.bind("<Button-1>", lambda e: webbrowser.open("https://t.me/nurlan_nursultan"))

        # 2. Main Content
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=40, pady=30)

        # - Top Section: Interactive Waveform
        self.waveform = WaveformWidget(self.main_container, on_seek=self._seek_playback, on_crop=self._apply_crop)
        self.waveform.frame.pack(fill="x", pady=(0, 25))

        # - Premium Control Panel
        self.control_panel = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.control_panel.pack(fill="x", pady=(0, 25))

        # Left Group: File Ops
        file_container = ctk.CTkFrame(self.control_panel, fg_color=BG_SURFACE, corner_radius=15, border_width=1, border_color=BORDER_COL)
        file_container.pack(side="left", padx=(0, 15))
        
        ctk.CTkButton(file_container, text="📁 Выбрать файл", width=160, height=52, corner_radius=10, 
                       fg_color="#2C2C2E", hover_color="#3A3A3C", font=ctk.CTkFont(weight="bold"), 
                       command=self._select_file).pack(padx=15, pady=15)

        # Center Group: Playback & Recording (Unified)
        audio_container = ctk.CTkFrame(self.control_panel, fg_color=BG_SURFACE, corner_radius=15, border_width=1, border_color=BORDER_COL)
        audio_container.pack(side="left", fill="both", expand=True)

        self.btn_play = ctk.CTkButton(audio_container, text="▶", width=60, height=60, corner_radius=30, 
                                       fg_color=ACCENT_BLUE, hover_color="#005FCC", font=ctk.CTkFont(size=24), 
                                       command=self._toggle_playback)
        self.btn_play.pack(side="left", padx=(20, 10), pady=12)

        self.btn_record = ctk.CTkButton(audio_container, text="🎤", width=60, height=60, corner_radius=30, 
                                         fg_color=ACCENT_RED, hover_color="#CC2D23", font=ctk.CTkFont(size=24), 
                                         command=self._toggle_recording)
        self.btn_record.pack(side="left", padx=10, pady=12)

        ctk.CTkButton(audio_container, text="📂", width=44, height=44, corner_radius=12, 
                       fg_color="#2C2C2E", hover_color="#3A3A3C", font=ctk.CTkFont(size=18), 
                       command=self._open_recordings).pack(side="left", padx=15)

        self.lbl_time = ctk.CTkLabel(audio_container, text="00:00 / 00:00", font=ctk.CTkFont(family="SF Pro Display", size=16, weight="bold"), text_color=TEXT_PRIMARY)
        self.lbl_time.pack(side="right", padx=25)

        # Right Group: Audio Enhancer
        util_container = ctk.CTkFrame(self.control_panel, fg_color=BG_SURFACE, corner_radius=15, border_width=1, border_color=BORDER_COL)
        util_container.pack(side="right", padx=(15, 0))
        
        ctk.CTkButton(util_container, text="⚡ Усилить", width=120, height=52, corner_radius=10, 
                       fg_color="#2C2C2E", hover_color="#3A3A3C", font=ctk.CTkFont(weight="bold"), 
                       command=self._normalize_audio).pack(padx=15, pady=15)

        # 3. Transcription Box (Immersive)
        trans_container = ctk.CTkFrame(self.main_container, fg_color=BG_SURFACE, corner_radius=20, border_width=1, border_color=BORDER_COL)
        trans_container.pack(fill="both", expand=True)

        trans_head = ctk.CTkFrame(trans_container, fg_color="transparent", height=45)
        trans_head.pack(fill="x", side="top", padx=25, pady=(20, 0))

        ctk.CTkLabel(trans_head, text="Распознанный текст", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        ctk.CTkLabel(trans_head, text="API: Whisper-Turbo KSC2 CT2", font=ctk.CTkFont(size=12), text_color=ACCENT_BLUE).pack(side="right")

        self.text_output = ctk.CTkTextbox(trans_container, fg_color="transparent", font=("Times New Roman", 18), 
                                           text_color="#E8E8E8", padx=25, pady=25, wrap="word")
        self.text_output.pack(fill="both", expand=True)
        self.text_output.insert("1.0", "Готов к распознаванию...")

        # 4. Global Actions
        actions = ctk.CTkFrame(self.main_container, fg_color="transparent")
        actions.pack(fill="x", pady=(30, 0))

        self.btn_run = ctk.CTkButton(actions, text="🚀 Транскрибировать", width=300, height=56, corner_radius=15, 
                                      fg_color=ACCENT_BLUE, hover_color="#005FCC", font=ctk.CTkFont(size=18, weight="bold"), 
                                      command=self._start_transcription)
        self.btn_run.pack(side="left")

        ctk.CTkButton(actions, text="📋 Копировать", width=140, height=56, corner_radius=15, 
                       fg_color="#2C2C2E", hover_color="#3A3A3C", font=ctk.CTkFont(weight="bold"), 
                       command=self._copy_text).pack(side="right", padx=(15, 0))
        
        ctk.CTkButton(actions, text="💾 Сохранить", width=140, height=56, corner_radius=15, 
                       fg_color="#2C2C2E", hover_color="#3A3A3C", font=ctk.CTkFont(weight="bold"), 
                       command=self._save_text).pack(side="right")

        self.btn_reset = ctk.CTkButton(actions, text="🗑 Сброс", width=100, height=56, corner_radius=15, 
                                        fg_color="#3D1A1A", hover_color="#5D2A2A", font=ctk.CTkFont(weight="bold"), 
                                        command=self._reset_app)
        self.btn_reset.pack(side="right", padx=(0, 15))

    def _setup_menu(self):
        self.menubar = tk.Menu(self)
        self.configure(menu=self.menubar)
        
        # 1. Models Menu
        self.models_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Модели", menu=self.models_menu)
        self._refresh_models_menu()

    def _refresh_models_menu(self):
        self.models_menu.delete(0, "end")
        
        active = self.engine.current_model_name
        self.models_menu.add_command(label="🔄 Обновить список моделей", command=self._refresh_models_menu)
        self.models_menu.add_separator()
        
        # Detect local models in 'models' folder
        downloaded = []
        if os.path.exists(self.engine.models_dir):
            for d in sorted(os.listdir(self.engine.models_dir)):
                if os.path.isdir(os.path.join(self.engine.models_dir, d)) and not d.startswith(".locks"):
                    downloaded.append(d)
        
        if downloaded:
            for m in downloaded:
                prefix = "✅ " if m == active else "      "
                self.models_menu.add_command(label=f"{prefix}{m}", command=lambda x=m: self._change_model(x))
            self.models_menu.add_separator()
        
        # Standard models submenu
        dl_menu = tk.Menu(self.models_menu, tearoff=0)
        self.models_menu.add_cascade(label="Скачать стандартные (HF)", menu=dl_menu)
        
        for m in ["tiny", "base", "small", "medium", "large-v3", "turbo"]:
            self.models_menu.add_command(label=f"  ⬇️ {m}", command=lambda x=m: self._change_model(x))

    def _change_model(self, name):
        if self.is_transcribing:
            messagebox.showwarning("Внимание", "Нельзя менять модель во время транскрибации.")
            return
        
        self._update_status(f"⏳ Подготовка {name}...")
        self.btn_run.configure(state="disabled")
        
        def run():
            try:
                self.engine.load_model(model_name=name, callback=lambda msg, p: self.after(0, self._update_status, msg))
                self.after(0, self._update_status, f"✅ Активна: {name}")
                self.after(0, self._refresh_models_menu)
                self.after(0, lambda: messagebox.showinfo("Готово", f"Модель {name} активирована и готова."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
                self.after(0, self._update_status, "❌ Ошибка")
            finally:
                self.after(0, lambda: self.btn_run.configure(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    # ================================================================
    # ENGINE & IO
    # ================================================================
    def _select_file(self):
        file = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.m4a *.flac *.ogg")])
        if file: self._load_audio(file)

    def _load_audio(self, path):
        try:
            self._update_status("⏳ Оптимизированная загрузка...")
            # Use soundfile for much faster loading of standard files
            try:
                data, sr = sf.read(path)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1) # Mono conversion
                if sr != 16000:
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                self.sr = 16000
            except:
                # Fallback to librosa
                data, _ = librosa.load(path, sr=16000)
            
            self.audio_data = data
            self.waveform.show_audio(data, 16000)
            self.playback_pos = 0.0
            self._refresh_ui_pos()
            self._update_status(f"✅ Файл: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Не удалось загрузить: {e}")
            self._update_status("❌ Ошибка загрузки")

    def _toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.btn_record.configure(fg_color=ACCENT_RED, text="🎤")
            if self._recording_timer_id:
                self.after_cancel(self._recording_timer_id)
                self._recording_timer_id = None
                
            if hasattr(self, '_rec_stream'):
                try:
                    self._rec_stream.stop()
                    self._rec_stream.close()
                except: pass
                
            if self._rec_buffer:
                recorded = np.concatenate(self._rec_buffer)
                if len(recorded.shape) > 1: recorded = recorded[:, 0]
                self.audio_data = recorded.astype(np.float32)
                
                filename = f"rec_{datetime.now().strftime('%H%M%S')}.wav"
                rec_path = os.path.join(self.recordings_dir, filename)
                sf.write(rec_path, self.audio_data, 16000)
                
                self.waveform.show_audio(self.audio_data, 16000)
                self.playback_pos = 0.0
                self._refresh_ui_pos()
                self._update_status(f"✅ Записано: {filename}")
        else:
            try:
                self._rec_buffer = []
                self.record_seconds = 0
                self.is_recording = True
                self.waveform.audio_data = None # Clear old waveform
                self.waveform.live_rms = []
                self.btn_record.configure(fg_color="#333", text="⏹")
                self._update_status("🔴 Идет запись...")
                
                def cb(data, frames, time, status): 
                    self._rec_buffer.append(data.copy())
                    # Pre-calculate RMS on the audio thread to save UI thread time
                    rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
                    self.after(0, lambda r=rms: self.waveform.push_live(r))
                
                # Using 1600 blocksize (0.1s) to throttle UI updates and prevent freezes
                self._rec_stream = sd.InputStream(samplerate=16000, 
                                                  channels=1, 
                                                  callback=cb, 
                                                  blocksize=1600)
                self._rec_stream.start()
                self._update_record_timer()
            except Exception as e:
                self.is_recording = False
                self._update_status("❌ Ошибка микрофона")
                messagebox.showerror("Ошибка", f"Не удалось начать запись: {e}")

    def _update_record_timer(self):
        if self.is_recording:
            self.record_seconds += 1
            mins = self.record_seconds // 60
            secs = self.record_seconds % 60
            self.lbl_time.configure(text=f"{mins:02d}:{secs:02d} / 🔴 REC")
            self._recording_timer_id = self.after(1000, self._update_record_timer)

    def _toggle_playback(self):
        if self.is_playing: self.stop_playback()
        else: self.start_playback()

    def start_playback(self):
        if self.audio_data is None: return
        self.is_playing = True
        self.btn_play.configure(text="⏸")
        self.stop_playback_event.clear()

        def play_thread():
            try:
                def cb(outdata, frames, time_val, status):
                    if self.stop_playback_event.is_set(): raise sd.CallbackStop()
                    s = int(self.playback_pos)
                    curr_frames = min(frames, len(self.audio_data) - s)
                    if curr_frames <= 0: raise sd.CallbackStop()
                    outdata[:curr_frames, 0] = self.audio_data[s:s+curr_frames]
                    if curr_frames < frames: outdata[curr_frames:, 0] = 0
                    self.playback_pos += curr_frames

                with sd.OutputStream(samplerate=16000, channels=1, dtype='float32', callback=cb):
                    while self.is_playing and not self.stop_playback_event.is_set():
                        sd.sleep(100)
                        self.after(0, self._refresh_ui_pos)
                        if self.playback_pos >= len(self.audio_data): break
            except Exception: pass
            finally: self.after(0, self._on_playback_end)

        threading.Thread(target=play_thread, daemon=True).start()

    def _on_playback_end(self):
        self.is_playing = False
        self.btn_play.configure(text="▶")
        if self.playback_pos >= (len(self.audio_data) if self.audio_data is not None else 0):
            self.playback_pos = 0.0
        self._refresh_ui_pos()

    def stop_playback(self):
        self.is_playing = False
        self.stop_playback_event.set()

    def _refresh_ui_pos(self):
        if self.audio_data is None: return
        frac = self.playback_pos / len(self.audio_data)
        self.waveform.set_progress(frac)
        curr = int(self.playback_pos / 16000)
        total = int(len(self.audio_data) / 16000)
        self.lbl_time.configure(text=f"{curr//60:02d}:{curr%60:02d} / {total//60:02d}:{total%60:02d}")

    def _seek_playback(self, frac):
        if self.audio_data is None: return
        self.playback_pos = frac * len(self.audio_data)
        self._refresh_ui_pos()

    def _normalize_audio(self):
        if self.audio_data is None: return
        peak = np.max(np.abs(self.audio_data))
        if peak > 0:
            self.audio_data = self.audio_data / peak
            self.waveform.show_audio(self.audio_data, 16000)
            self._update_status("⚡ Звук нормализован")

    def _apply_crop(self, sub, sr):
        self.audio_data = sub
        self.waveform.show_audio(sub, 16000)
        self.playback_pos = 0.0
        self._refresh_ui_pos()

    def _open_recordings(self):
        if sys.platform == "darwin":
            os.system(f"open '{self.recordings_dir}'")
        else:
            webbrowser.open(self.recordings_dir)

    def _reset_app(self):
        self.stop_playback()
        self.audio_data = None
        self.playback_pos = 0.0
        self.waveform.canvas.delete("all")
        self.waveform.audio_data = None
        self.text_output.delete("1.0", "end")
        self.text_output.insert("1.0", "Готов к распознаванию...")
        self._refresh_ui_pos()
        self._update_status("🔄 Сброшено")

    def _start_transcription(self):
        if self.audio_data is None:
            messagebox.showwarning("Внимание", "Загрузите аудиофайл перед транскрибацией.")
            return
        
        if self.is_transcribing:
            self.stop_trans_event.set()
            self._update_status("🛑 Остановка...")
            return

        self.is_transcribing = True
        self.stop_trans_event.clear()
        self.btn_run.configure(text="⏹ Остановить транскрипцию", fg_color=ACCENT_RED, hover_color="#CC2D23")
        
        # Flag to clear placeholder on first chunk
        self._first_trans_chunk = True
        
        def run():
            try:
                self.engine.load_model(callback=lambda msg, p: self.after(0, self._update_status, msg))
                
                def update_txt(t):
                    if hasattr(self, '_first_trans_chunk') and self._first_trans_chunk:
                        self.after(0, lambda: self.text_output.delete("1.0", "end"))
                        self._first_trans_chunk = False
                    self.after(0, lambda: self.text_output.insert("end", t))
                
                res = self.engine.transcribe(
                    self.audio_data, 
                    text_callback=update_txt,
                    stop_event=self.stop_trans_event,
                    progress_callback=lambda msg, p: self.after(0, self._update_status, msg)
                )
                
                if self.stop_trans_event.is_set():
                    self.after(0, self._update_status, "🛑 Прервано пользователем")
                else:
                    self.after(0, self._update_status, "✅ Готово")
                    
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.after(0, self._update_status, "❌ Ошибка")
            finally:
                self.is_transcribing = False
                self.after(0, lambda: self.btn_run.configure(text="🚀 Транскрибировать", fg_color=ACCENT_BLUE, hover_color="#005FCC"))

        threading.Thread(target=run, daemon=True).start()

    def _copy_text(self):
        self.clipboard_clear(); self.clipboard_append(self.text_output.get("1.0", "end-1c"))
        self._update_status("📋 Текст скопирован")

    def _save_text(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            with open(path, "w") as f: f.write(self.text_output.get("1.0", "end-1c"))
            self._update_status("💾 Сохранено")

    def _update_status(self, msg): self.lbl_status.configure(text=msg)

if __name__ == "__main__":
    app = QazTriberApp()
    app.mainloop()