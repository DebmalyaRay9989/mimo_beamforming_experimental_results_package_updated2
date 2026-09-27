"""
╔══════════════════════════════════════════════════════════════════════╗
║   MIMO BEAMFORMING OPTIMIZER  v2.0  —  Deep RL Edition              ║
║   DQN · Double DQN · Dueling DQN · Multi-Agent · Optuna Tuning      ║
║   Enhanced GUI: Tabbed layout, live SNR sweep, BER/capacity,        ║
║   per-antenna heatmap, session compare, export suite, theme engine  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import threading
import csv
import json
import os
import pickle
import io
import copy
from datetime import datetime
from fpdf import FPDF
import pandas as pd
import optuna
import time
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import traceback
import math
import sys

# ── Reproducibility ──────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS & DEFAULTS
# ──────────────────────────────────────────────────────────────────────
PRESETS = {
    "Low":    {"num_antennas":2,"num_beams":8, "snr_db":5, "episodes":500, "batch_size":16,
               "gamma":0.90,"epsilon":1.0,"epsilon_min":0.01,"epsilon_decay":0.990,"learning_rate":0.001},
    "Medium": {"num_antennas":4,"num_beams":16,"snr_db":10,"episodes":1000,"batch_size":32,
               "gamma":0.95,"epsilon":1.0,"epsilon_min":0.01,"epsilon_decay":0.995,"learning_rate":0.001},
    "High":   {"num_antennas":8,"num_beams":32,"snr_db":15,"episodes":2000,"batch_size":64,
               "gamma":0.98,"epsilon":1.0,"epsilon_min":0.01,"epsilon_decay":0.997,"learning_rate":0.0005},
    "Massive":{"num_antennas":16,"num_beams":64,"snr_db":20,"episodes":3000,"batch_size":128,
               "gamma":0.99,"epsilon":1.0,"epsilon_min":0.005,"epsilon_decay":0.998,"learning_rate":0.0003},
}

THEMES = {
    "Dark":    {"bg":"#0d1117","bg2":"#161b22","bg3":"#21262d","accent":"#58a6ff",
                "accent2":"#3fb950","warn":"#d29922","err":"#f85149",
                "fg":"#e6edf3","fg2":"#8b949e","border":"#30363d",
                "plot_bg":"#0d1117","text":"white","canvas_bg":"#161b22"},
    "Light":   {"bg":"#f6f8fa","bg2":"#ffffff","bg3":"#eaeef2","accent":"#0969da",
                "accent2":"#1a7f37","warn":"#9a6700","err":"#cf222e",
                "fg":"#1f2328","fg2":"#656d76","border":"#d0d7de",
                "plot_bg":"#ffffff","text":"black","canvas_bg":"#f6f8fa"},
    "Synthwave":{"bg":"#0a0010","bg2":"#120020","bg3":"#1a0030","accent":"#ff00ff",
                 "accent2":"#00ffff","warn":"#ffaa00","err":"#ff3366",
                 "fg":"#f0e0ff","fg2":"#a070c0","border":"#4a0060",
                 "plot_bg":"#0a0010","text":"#f0e0ff","canvas_bg":"#120020"},
    "Ocean":   {"bg":"#001825","bg2":"#002535","bg3":"#003045","accent":"#00b4d8",
                "accent2":"#90e0ef","warn":"#f4a261","err":"#e63946",
                "fg":"#caf0f8","fg2":"#90e0ef","border":"#014f86",
                "plot_bg":"#001825","text":"#caf0f8","canvas_bg":"#002535"},
}

# ──────────────────────────────────────────────────────────────────────
# MIMO ENVIRONMENT  (extended with BER + capacity metrics)
# ──────────────────────────────────────────────────────────────────────
class MIMOEnvironment:
    def __init__(self, num_antennas=4, num_beams=16, snr_db=10,
                 channel_model="Rayleigh", codebook_type="DFT",
                 mobility=0.05, frequency_ghz=2.4, bandwidth_mhz=10.0):
        self.num_antennas  = num_antennas
        self.num_beams     = num_beams
        self.snr_db        = snr_db
        self.snr           = 10 ** (snr_db / 10.0)
        self.channel_model = channel_model
        self.mobility      = mobility          # channel variation per step
        self.frequency_ghz = frequency_ghz
        self.bandwidth_mhz = bandwidth_mhz
        self.codebook      = self._generate_codebook(codebook_type)
        self.channel       = None
        self.state_size    = num_antennas * 2
        self.action_size   = num_beams
        self.step_count    = 0
        self._history_reward = []
        self._history_rate   = []

    # ── Codebook ─────────────────────────────────────────────────────
    def _generate_codebook(self, codebook_type):
        if codebook_type == "Random":
            cb = (np.random.randn(self.num_beams, self.num_antennas) +
                  1j * np.random.randn(self.num_beams, self.num_antennas))
            return cb / np.linalg.norm(cb, axis=1, keepdims=True)
        elif codebook_type == "Zadoff-Chu":
            # Zadoff-Chu sequences — good cross-correlation properties
            u = 1
            cb = np.zeros((self.num_beams, self.num_antennas), dtype=complex)
            for beam_idx in range(self.num_beams):
                offset = (2 * np.pi * beam_idx) / self.num_beams
                for n in range(self.num_antennas):
                    cb[beam_idx, n] = np.exp(-1j * np.pi * u * n * (n + 1) / self.num_antennas + 1j * offset)
            return cb / np.linalg.norm(cb, axis=1, keepdims=True)
        else:  # DFT
            angles = np.linspace(0, 2 * np.pi, self.num_beams)
            cb = np.zeros((self.num_beams, self.num_antennas), dtype=complex)
            for i, theta in enumerate(angles):
                cb[i] = np.exp(1j * np.arange(self.num_antennas) * theta) / np.sqrt(self.num_antennas)
            return cb

    # ── Channel reset ─────────────────────────────────────────────────
    def reset(self):
        self.step_count = 0
        if self.channel_model == "Rician":
            LoS  = np.exp(1j * np.linspace(0, 2 * np.pi, self.num_antennas))
            NLoS = (np.random.randn(self.num_antennas) +
                    1j * np.random.randn(self.num_antennas)) / np.sqrt(2)
            K = 5
            self.channel = np.sqrt(K / (K + 1)) * LoS + np.sqrt(1 / (K + 1)) * NLoS
        elif self.channel_model == "Saleh-Valenzuela":
            # Cluster-based ray channel
            num_clusters, num_rays = 3, 4
            h = np.zeros(self.num_antennas, dtype=complex)
            for _ in range(num_clusters):
                cluster_gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                for _ in range(num_rays):
                    aoa = np.random.uniform(0, 2 * np.pi)
                    steering = np.exp(1j * np.pi * np.arange(self.num_antennas) * np.sin(aoa))
                    ray_gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2 * num_rays)
                    h += cluster_gain * ray_gain * steering
            self.channel = h
        elif self.channel_model == "Custom":
            if self.channel is None:
                self.channel = (np.random.randn(self.num_antennas) +
                                1j * np.random.randn(self.num_antennas)) / np.sqrt(2)
        else:  # Rayleigh
            self.channel = (np.random.randn(self.num_antennas) +
                            1j * np.random.randn(self.num_antennas)) / np.sqrt(2)

        if self.channel is not None:
            norm = np.linalg.norm(self.channel)
            if norm > 1e-9:
                self.channel /= norm

        return np.concatenate([self.channel.real, self.channel.imag])

    # ── Environment step ──────────────────────────────────────────────
    def step(self, action):
        self.step_count += 1
        beam_vector  = self.codebook[action]
        signal_power = np.abs(np.dot(self.channel.conj(), beam_vector)) ** 2
        rate         = np.log2(1 + self.snr * signal_power)

        # Shannon capacity with bandwidth
        capacity_mbps = self.bandwidth_mhz * rate

        # BER estimate for BPSK in AWGN (approximate)
        snr_effective = self.snr * signal_power
        ber = 0.5 * math.erfc(np.sqrt(max(snr_effective, 1e-12)))

        # Spectral efficiency (reward)
        reward = rate

        # Channel evolution
        noise = (np.random.randn(self.num_antennas) +
                 1j * np.random.randn(self.num_antennas)) / np.sqrt(2)
        self.channel += self.mobility * noise
        norm = np.linalg.norm(self.channel)
        if norm > 1e-9:
            self.channel /= norm

        next_state = np.concatenate([self.channel.real, self.channel.imag])

        self._history_reward.append(reward)
        self._history_rate.append(rate)

        return next_state, reward, False, {
            "rate": rate, "beam_vector": beam_vector,
            "capacity_mbps": capacity_mbps, "ber": ber,
            "signal_power": signal_power,
            "beam_idx": action,
        }

    # ── Utility: best beam by exhaustive search ───────────────────────
    def optimal_beam(self):
        powers = [np.abs(np.dot(self.channel.conj(), self.codebook[i])) ** 2
                  for i in range(self.num_beams)]
        best = int(np.argmax(powers))
        return best, np.log2(1 + self.snr * powers[best])

    # ── SNR sweep: compute average rate across SNR range ─────────────
    def snr_sweep(self, snr_range_db, agent, num_episodes=20):
        orig_snr = self.snr_db
        results  = []
        for snr_db in snr_range_db:
            self.snr_db = snr_db
            self.snr    = 10 ** (snr_db / 10.0)
            rates = []
            for _ in range(num_episodes):
                state = self.reset()
                _, _, _, info = self.step(agent.act_greedy(state))
                rates.append(info["rate"])
            results.append(np.mean(rates))
        # Restore
        self.snr_db = orig_snr
        self.snr    = 10 ** (orig_snr / 10.0)
        return results


# ──────────────────────────────────────────────────────────────────────
# DQN AGENT  (+ Double / Dueling, greedy eval, weight export)
# ──────────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001,
                 algo="DQN", hidden_units=64, dropout_rate=0.0,
                 use_batch_norm=False):
        self.state_size    = state_size
        self.action_size   = action_size
        self.memory        = deque(maxlen=5000)
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.algo          = algo
        self.hidden_units  = hidden_units
        self.dropout_rate  = dropout_rate
        self.use_batch_norm= use_batch_norm
        self.train_step    = 0
        self.q_history     = []          # mean Q per replay
        self.loss_history  = []

        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        hu = self.hidden_units
        if "Dueling" in self.algo:
            inputs = Input(shape=(self.state_size,))
            x = Dense(hu, activation='relu')(inputs)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            x = Dense(hu // 2, activation='relu')(x)
            value     = Dense(1)(x)
            advantage = Dense(self.action_size)(x)
            mean_adv  = tf.reduce_mean(advantage, axis=1, keepdims=True)
            q_values  = value + (advantage - mean_adv)
            model = Model(inputs=inputs, outputs=q_values)
        else:
            layers = [Dense(hu, input_dim=self.state_size, activation='relu')]
            if self.use_batch_norm:
                layers.append(BatchNormalization())
            if self.dropout_rate > 0:
                layers.append(Dropout(self.dropout_rate))
            layers.append(Dense(hu // 2, activation='relu'))
            layers.append(Dense(self.action_size, activation='linear'))
            model = Sequential(layers)

        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return self.act_greedy(state)

    def act_greedy(self, state):
        s = np.reshape(state, [1, self.state_size])
        q = self.model.predict(s, verbose=0)
        return int(np.argmax(q[0]))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
        minibatch   = random.sample(self.memory, batch_size)
        states      = np.array([e[0] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        targets     = self.model.predict(states, verbose=0)
        next_q      = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            elif self.algo == "Double DQN":
                best = int(np.argmax(self.model.predict(next_states, verbose=0)[i]))
                targets[i][action] = reward + self.gamma * next_q[i][best]
            else:
                targets[i][action] = reward + self.gamma * np.amax(next_q[i])

        hist = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        loss = float(hist.history["loss"][0])
        self.loss_history.append(loss)

        mean_q = float(np.mean(targets))
        self.q_history.append(mean_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_step += 1
        return loss

    def get_q_table_sample(self, env):
        """Sample Q-values across all codebook beams for current channel."""
        if env.channel is None:
            return None
        state = np.concatenate([env.channel.real, env.channel.imag])
        s = np.reshape(state, [1, self.state_size])
        return self.model.predict(s, verbose=0)[0]

    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename)
        self.update_target_model()

    def export_weights_csv(self, filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            for layer in self.model.layers:
                for w in layer.get_weights():
                    writer.writerow([layer.name] + w.flatten().tolist())

    def get_summary_dict(self):
        return {
            "algo": self.algo,
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "epsilon": round(self.epsilon, 5),
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "memory_size": len(self.memory),
            "train_steps": self.train_step,
        }


# ──────────────────────────────────────────────────────────────────────
# SESSION RECORD  (stores a completed run for comparison)
# ──────────────────────────────────────────────────────────────────────
class SessionRecord:
    def __init__(self, label, params, rewards, rates, losses, q_values,
                 ber_history=None, capacity_history=None):
        self.label            = label
        self.params           = params
        self.rewards          = rewards
        self.rates            = rates
        self.losses           = losses
        self.q_values         = q_values
        self.ber_history      = ber_history or {}
        self.capacity_history = capacity_history or {}
        self.timestamp        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def summary(self):
        algo = next(iter(self.rates), "?")
        r = self.rates.get(algo, [])
        return {
            "label":    self.label,
            "algo":     algo,
            "episodes": len(r),
            "avg_rate": f"{np.mean(r):.3f}" if r else "N/A",
            "max_rate": f"{max(r):.3f}" if r else "N/A",
            "timestamp":self.timestamp,
        }


# ──────────────────────────────────────────────────────────────────────
# MAIN GUI APPLICATION
# ──────────────────────────────────────────────────────────────────────
class MIMOBeamformingGUI:
    # ── Init ──────────────────────────────────────────────────────────
    def __init__(self, root):
        self.root = root
        self.root.title("MIMO Beamforming Optimizer  v2.0")
        self.root.geometry("1600x960")
        self.root.minsize(1100, 700)

        # State
        self.current_theme    = "Dark"
        self.T                = THEMES[self.current_theme]
        self.training_thread  = None
        self.stop_training    = False
        self.pause_training   = False
        self.agent            = None
        self.env              = None
        self.rewards          = {}
        self.rates            = {}
        self.losses           = {}
        self.q_values         = {}
        self.ber_history      = {}
        self.capacity_history = {}
        self.current_beam     = None
        self.current_channel  = None
        self.log_messages     = []
        self.training_state   = None
        self.param_lock       = threading.Lock()
        self.smoothing_window = 50
        self.batch_configs    = []
        self.start_time       = None
        self.sessions         = []          # saved SessionRecord list
        self.snr_sweep_result = None
        self.tooltip          = None
        self.training_step_count = 0

        self.plot_config = {
            "rewards":True,"rates":True,"beam_pattern":True,
            "channel_state":True,"channel_3d":False,"beam_heatmap":False,
            "loss_curve":True,"q_value":True,"ber_curve":False,"capacity":False,
        }

        # Matplotlib figures  (main + SNR sweep + compare)
        self.fig  = plt.figure(figsize=(12, 7))
        self.fig2 = plt.figure(figsize=(8, 4))     # SNR sweep
        self.fig3 = plt.figure(figsize=(10, 5))    # session compare

        self._build_ui()
        self._apply_full_theme()
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    # ─────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._configure_styles()

        # Root layout: left sidebar + right notebook
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ── Sidebar ──────────────────────────────────────────────────
        self.sidebar = tk.Frame(self.root, width=280)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=(8,0), pady=8)
        self.sidebar.grid_propagate(False)
        self._build_sidebar()

        # ── Right area: notebook ──────────────────────────────────────
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        self.tab_train   = ttk.Frame(self.notebook)
        self.tab_analysis= ttk.Frame(self.notebook)
        self.tab_sweep   = ttk.Frame(self.notebook)
        self.tab_compare = ttk.Frame(self.notebook)
        self.tab_arch    = ttk.Frame(self.notebook)
        self.tab_about   = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_train,    text="  Training  ")
        self.notebook.add(self.tab_analysis, text="  Analysis  ")
        self.notebook.add(self.tab_sweep,    text="  SNR Sweep  ")
        self.notebook.add(self.tab_compare,  text="  Compare  ")
        self.notebook.add(self.tab_arch,     text="  Architecture  ")
        self.notebook.add(self.tab_about,    text="  About  ")

        self._build_tab_train()
        self._build_tab_analysis()
        self._build_tab_sweep()
        self._build_tab_compare()
        self._build_tab_arch()
        self._build_tab_about()

    # ── Sidebar ───────────────────────────────────────────────────────
    def _build_sidebar(self):
        T = self.T

        # Title
        title_fr = tk.Frame(self.sidebar)
        title_fr.pack(fill="x", padx=8, pady=(10,4))
        tk.Label(title_fr, text="MIMO BF Optimizer",
                 font=("Courier New", 12, "bold")).pack(side="left")

        # Theme selector
        theme_fr = tk.Frame(self.sidebar)
        theme_fr.pack(fill="x", padx=8, pady=2)
        tk.Label(theme_fr, text="Theme:", font=("Segoe UI", 9)).pack(side="left")
        self.theme_combo = ttk.Combobox(theme_fr, values=list(THEMES.keys()),
                                        width=10, state="readonly")
        self.theme_combo.set(self.current_theme)
        self.theme_combo.pack(side="left", padx=4)
        self.theme_combo.bind("<<ComboboxSelected>>", self._on_theme_change)

        sep = ttk.Separator(self.sidebar, orient="horizontal")
        sep.pack(fill="x", padx=8, pady=6)

        # Scrollable param area
        scroll_canvas = tk.Canvas(self.sidebar, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.sidebar, orient="vertical",
                                  command=scroll_canvas.yview)
        self.param_scroll_frame = tk.Frame(scroll_canvas)
        self.param_scroll_frame.bind("<Configure>",
            lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.create_window((0,0), window=self.param_scroll_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build_param_panel(self.param_scroll_frame)

    def _build_param_panel(self, parent):
        # ── MIMO Parameters ─────────────────────────────────────────
        cfg = ttk.LabelFrame(parent, text="Configuration", padding=8)
        cfg.pack(fill="x", padx=6, pady=4)

        self.params = {
            "num_antennas": {"label":"Antennas","default":4,"type":int,"min":1,"max":64,
                             "tooltip":"Number of transmit antennas"},
            "num_beams":    {"label":"Beams","default":16,"type":int,"min":2,"max":256,
                             "tooltip":"Size of the beamforming codebook"},
            "snr_db":       {"label":"SNR (dB)","default":10,"type":float,"min":-20,"max":40,
                             "tooltip":"Signal-to-Noise Ratio in dB"},
            "episodes":     {"label":"Episodes","default":1000,"type":int,"min":1,"max":20000,
                             "tooltip":"Number of training episodes"},
            "batch_size":   {"label":"Batch Size","default":32,"type":int,"min":4,"max":512,
                             "tooltip":"Replay buffer minibatch size"},
            "gamma":        {"label":"Discount γ","default":0.95,"type":float,"min":0.0,"max":1.0,
                             "tooltip":"Future reward discount factor"},
            "epsilon":      {"label":"Epsilon ε","default":1.0,"type":float,"min":0.0,"max":1.0,
                             "tooltip":"Initial exploration rate"},
            "epsilon_min":  {"label":"Min ε","default":0.01,"type":float,"min":0.0,"max":1.0,
                             "tooltip":"Minimum exploration rate"},
            "epsilon_decay":{"label":"ε Decay","default":0.995,"type":float,"min":0.8,"max":1.0,
                             "tooltip":"Exploration rate decay per step"},
            "learning_rate":{"label":"LR","default":0.001,"type":float,"min":1e-5,"max":0.1,
                             "tooltip":"Adam optimizer learning rate"},
        }
        self.entries = {}

        for key, param in self.params.items():
            row = tk.Frame(cfg)
            row.pack(fill="x", pady=2)
            lbl = tk.Label(row, text=param["label"], width=13, anchor="w",
                           font=("Segoe UI", 9))
            lbl.pack(side="left")
            entry = ttk.Entry(row, width=9)
            entry.insert(0, str(param["default"]))
            entry.pack(side="left", padx=3)
            self.entries[key] = entry

            # Tooltip
            tip_lbl = tk.Label(row, text="ⓘ", font=("Segoe UI", 8), cursor="hand2")
            tip_lbl.pack(side="left")
            tip_lbl.bind("<Enter>", lambda e, t=param["tooltip"]: self._show_tooltip(e, t))
            tip_lbl.bind("<Leave>", self._hide_tooltip)

        # ── Dropdowns ───────────────────────────────────────────────
        dd = ttk.LabelFrame(parent, text="Model & Channel", padding=8)
        dd.pack(fill="x", padx=6, pady=4)

        for lbl_text, attr, values, default in [
            ("RL Algorithm",  "algo_combo",     ["DQN","Double DQN","Dueling DQN"], "DQN"),
            ("Channel Model", "channel_combo",  ["Rayleigh","Rician","Saleh-Valenzuela","Custom"], "Rayleigh"),
            ("Codebook",      "codebook_combo", ["DFT","Random","Zadoff-Chu"], "DFT"),
        ]:
            tk.Label(dd, text=lbl_text, font=("Segoe UI", 9), anchor="w").pack(
                anchor="w", padx=2, pady=(4,0))
            combo = ttk.Combobox(dd, values=values, width=18, state="readonly")
            combo.set(default)
            combo.pack(fill="x", padx=2, pady=(0,2))
            setattr(self, attr, combo)

        # ── Advanced Architecture ────────────────────────────────────
        adv = ttk.LabelFrame(parent, text="Architecture", padding=8)
        adv.pack(fill="x", padx=6, pady=4)

        for lbl_text, attr, default in [
            ("Hidden Units", "hidden_units_entry", "64"),
            ("Dropout Rate", "dropout_entry",      "0.0"),
            ("Mobility",     "mobility_entry",     "0.05"),
            ("BW (MHz)",     "bandwidth_entry",    "10.0"),
        ]:
            row = tk.Frame(adv)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=lbl_text, width=13, anchor="w",
                     font=("Segoe UI",9)).pack(side="left")
            e = ttk.Entry(row, width=9)
            e.insert(0, default)
            e.pack(side="left", padx=3)
            setattr(self, attr, e)

        self.batch_norm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv, text="Batch Normalization",
                        variable=self.batch_norm_var).pack(anchor="w", pady=2)

        # ── Presets ──────────────────────────────────────────────────
        pre = ttk.LabelFrame(parent, text="Presets", padding=8)
        pre.pack(fill="x", padx=6, pady=4)
        self.preset_combo = ttk.Combobox(pre, values=list(PRESETS.keys()),
                                         width=18, state="readonly")
        self.preset_combo.pack(fill="x", pady=2)
        self.preset_combo.bind("<<ComboboxSelected>>", self._load_preset)
        ttk.Button(pre, text="Apply Params",
                   command=self._apply_params).pack(fill="x", pady=2)
        ttk.Button(pre, text="Upload Channel",
                   command=self._upload_channel).pack(fill="x", pady=2)
        ttk.Button(pre, text="Reset Defaults",
                   command=self._reset_defaults).pack(fill="x", pady=2)

        # ── Plot Options ─────────────────────────────────────────────
        plot_fr = ttk.LabelFrame(parent, text="Plot Layers", padding=8)
        plot_fr.pack(fill="x", padx=6, pady=4)
        self._plot_vars = {}
        plot_labels = {
            "rewards":"Rewards","rates":"Spectral Rate",
            "loss_curve":"Loss Curve","q_value":"Mean Q-Value",
            "beam_pattern":"Beam Pattern","channel_state":"Channel State",
            "ber_curve":"BER Curve","capacity":"Capacity (Mbps)",
            "channel_3d":"3-D Channel","beam_heatmap":"Beam Heatmap",
        }
        for key, label in plot_labels.items():
            var = tk.BooleanVar(value=self.plot_config.get(key, False))
            self._plot_vars[key] = var
            ttk.Checkbutton(plot_fr, text=label, variable=var,
                command=lambda k=key, v=var: self._toggle_plot(k, v.get())
            ).pack(anchor="w", padx=4)

        tk.Label(plot_fr, text="Smoothing Window", font=("Segoe UI",9)).pack(anchor="w", padx=4)
        self.smooth_slider = ttk.Scale(plot_fr, from_=5, to=200,
                                       orient="horizontal", command=self._update_smoothing)
        self.smooth_slider.set(self.smoothing_window)
        self.smooth_slider.pack(fill="x", padx=4, pady=2)

    # ── Training Tab ─────────────────────────────────────────────────
    def _build_tab_train(self):
        tab = self.tab_train
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # Plot area
        plot_fr = ttk.Frame(tab)
        plot_fr.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_fr)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tb_fr = ttk.Frame(plot_fr)
        tb_fr.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tb_fr)
        ttk.Button(tb_fr, text="💾 Save Plot",
                   command=self._save_plot).pack(side="left", padx=4)
        ttk.Button(tb_fr, text="📋 Copy Plot",
                   command=self._copy_plot_to_clipboard).pack(side="left", padx=4)

        # Control bar
        ctrl = ttk.Frame(tab)
        ctrl.grid(row=1, column=0, sticky="ew", padx=6, pady=2)
        self.train_button  = ttk.Button(ctrl, text="▶ Train",  command=self._start_training)
        self.pause_button  = ttk.Button(ctrl, text="⏸ Pause",  command=self._pause_training,  state="disabled")
        self.stop_button   = ttk.Button(ctrl, text="⏹ Stop",   command=self._stop_training,   state="disabled")
        self.resume_button = ttk.Button(ctrl, text="⏭ Resume", command=self._resume_training, state="disabled")
        self.save_button   = ttk.Button(ctrl, text="💾 Save",  command=self._save_model)
        self.load_button   = ttk.Button(ctrl, text="📂 Load",  command=self._load_model)
        self.test_button   = ttk.Button(ctrl, text="🧪 Test",  command=self._test_agent)
        self.report_button = ttk.Button(ctrl, text="📄 Export",command=self._export_report)
        self.opt_button    = ttk.Button(ctrl, text="🔧 Optuna",command=self._optimize_params)
        self.batch_button  = ttk.Button(ctrl, text="📦 Batch", command=self._batch_mode)
        self.session_btn   = ttk.Button(ctrl, text="💾 Save Session", command=self._save_session)

        for btn in [self.train_button, self.pause_button, self.stop_button,
                    self.resume_button, self.save_button, self.load_button,
                    self.test_button, self.report_button, self.opt_button,
                    self.batch_button, self.session_btn]:
            btn.pack(side="left", padx=3, pady=4)

        # Metrics dashboard
        met_fr = ttk.LabelFrame(tab, text="Live Metrics", padding=6)
        met_fr.grid(row=2, column=0, sticky="ew", padx=6, pady=2)
        self.status_label = ttk.Label(met_fr, text="Ready", foreground="gray")
        self.status_label.pack(anchor="w")
        mg = ttk.Frame(met_fr)
        mg.pack(fill="x")
        self.metrics_labels = {}
        metric_defs = [
            ("episode",       "Episode: —"),
            ("avg_reward",    "Avg Reward: —"),
            ("max_rate",      "Max Rate: —"),
            ("epsilon",       "ε: —"),
            ("loss",          "Loss: —"),
            ("ber",           "BER: —"),
            ("capacity",      "Cap: —"),
            ("memory",        "Mem: —"),
            ("training_time", "Time: —"),
        ]
        for col, (key, txt) in enumerate(metric_defs):
            lbl = ttk.Label(mg, text=txt, font=("Segoe UI", 9))
            lbl.grid(row=0, column=col, padx=6, pady=2, sticky="w")
            self.metrics_labels[key] = lbl

        self.progress = ttk.Progressbar(tab, mode="determinate")
        self.progress.grid(row=3, column=0, sticky="ew", padx=6, pady=2)

        # Log console
        log_fr = ttk.LabelFrame(tab, text="Console", padding=6)
        log_fr.grid(row=4, column=0, sticky="ew", padx=6, pady=4)
        log_inner = tk.Frame(log_fr)
        log_inner.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_inner, height=5, state="disabled",
                                font=("Courier New", 9), wrap="word")
        log_scroll = ttk.Scrollbar(log_inner, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        log_btn_fr = tk.Frame(log_fr)
        log_btn_fr.pack(fill="x")
        ttk.Button(log_btn_fr, text="Clear Log", command=self._clear_log).pack(side="left", padx=4)
        ttk.Button(log_btn_fr, text="Save Log",  command=self._save_log).pack(side="left", padx=4)

        # Animation for beam pattern
        self.ani = FuncAnimation(self.fig, self._animate,
                                 interval=1200, blit=False, cache_frame_data=False)

    # ── Analysis Tab ──────────────────────────────────────────────────
    def _build_tab_analysis(self):
        tab = self.tab_analysis
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)
        tab.grid_rowconfigure(2, weight=0)

        # Q-value bar chart frame
        top = ttk.LabelFrame(tab, text="Q-Value Distribution (current state)", padding=8)
        top.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)

        self.fig_analysis = plt.figure(figsize=(10, 5))
        self.canvas_analysis = FigureCanvasTkAgg(self.fig_analysis, master=top)
        self.canvas_analysis.get_tk_widget().pack(fill="both", expand=True)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        ttk.Button(ctrl, text="🔄 Refresh Q-Values", command=self._refresh_qvalues).pack(side="left", padx=4)
        ttk.Button(ctrl, text="📊 BER vs SNR",       command=self._plot_ber_vs_snr).pack(side="left", padx=4)
        ttk.Button(ctrl, text="📈 Capacity Curve",   command=self._plot_capacity_curve).pack(side="left", padx=4)
        ttk.Button(ctrl, text="🗺 Codebook Heatmap", command=self._plot_codebook_heatmap).pack(side="left", padx=4)
        ttk.Button(ctrl, text="📉 Convergence",      command=self._plot_convergence).pack(side="left", padx=4)

        # Stats table
        stats_fr = ttk.LabelFrame(tab, text="Session Statistics", padding=8)
        stats_fr.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
        cols = ("Metric", "Value")
        self.stats_tree = ttk.Treeview(stats_fr, columns=cols, show="headings", height=8)
        for c in cols:
            self.stats_tree.heading(c, text=c)
            self.stats_tree.column(c, width=200)
        self.stats_tree.pack(fill="x", expand=True)
        ttk.Button(stats_fr, text="Refresh Stats",
                   command=self._refresh_stats_table).pack(pady=4)

    # ── SNR Sweep Tab ─────────────────────────────────────────────────
    def _build_tab_sweep(self):
        tab = self.tab_sweep
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctrl = ttk.LabelFrame(tab, text="SNR Sweep Configuration", padding=8)
        ctrl.grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        tk.Label(ctrl, text="SNR Min (dB):", font=("Segoe UI",9)).grid(row=0,column=0,padx=6,pady=3,sticky="w")
        self.sweep_min = ttk.Entry(ctrl, width=8); self.sweep_min.insert(0,"-10"); self.sweep_min.grid(row=0,column=1,padx=4)
        tk.Label(ctrl, text="SNR Max (dB):", font=("Segoe UI",9)).grid(row=0,column=2,padx=6,pady=3,sticky="w")
        self.sweep_max = ttk.Entry(ctrl, width=8); self.sweep_max.insert(0,"30");  self.sweep_max.grid(row=0,column=3,padx=4)
        tk.Label(ctrl, text="Steps:", font=("Segoe UI",9)).grid(row=0,column=4,padx=6,pady=3,sticky="w")
        self.sweep_steps= ttk.Entry(ctrl, width=6); self.sweep_steps.insert(0,"20"); self.sweep_steps.grid(row=0,column=5,padx=4)
        tk.Label(ctrl, text="Episodes/pt:", font=("Segoe UI",9)).grid(row=0,column=6,padx=6,sticky="w")
        self.sweep_eps  = ttk.Entry(ctrl, width=6); self.sweep_eps.insert(0,"30"); self.sweep_eps.grid(row=0,column=7,padx=4)

        ttk.Button(ctrl, text="▶ Run Sweep",    command=self._run_snr_sweep).grid(row=0,column=8,padx=8)
        ttk.Button(ctrl, text="💾 Export Sweep",command=self._export_sweep).grid(row=0,column=9,padx=4)

        sweep_plot_fr = ttk.Frame(tab)
        sweep_plot_fr.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self.canvas_sweep = FigureCanvasTkAgg(self.fig2, master=sweep_plot_fr)
        self.canvas_sweep.get_tk_widget().pack(fill="both", expand=True)

    # ── Compare Tab ───────────────────────────────────────────────────
    def _build_tab_compare(self):
        tab = self.tab_compare
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        ttk.Button(ctrl, text="📊 Compare Sessions", command=self._compare_sessions).pack(side="left",padx=4)
        ttk.Button(ctrl, text="🗑 Clear Sessions",    command=self._clear_sessions).pack(side="left",padx=4)
        ttk.Button(ctrl, text="📄 Export Comparison", command=self._export_comparison).pack(side="left",padx=4)

        # Session list
        list_fr = ttk.LabelFrame(tab, text="Saved Sessions", padding=6)
        list_fr.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        cols = ("Label","Algo","Episodes","Avg Rate","Max Rate","Timestamp")
        self.session_tree = ttk.Treeview(list_fr, columns=cols,
                                         show="headings", height=6)
        for c in cols:
            self.session_tree.heading(c, text=c)
            self.session_tree.column(c, width=120)
        self.session_tree.pack(fill="x", expand=True)

        compare_fr = ttk.Frame(tab)
        compare_fr.grid(row=2, column=0, sticky="nsew", padx=6, pady=4)
        tab.grid_rowconfigure(2, weight=1)
        self.canvas_compare = FigureCanvasTkAgg(self.fig3, master=compare_fr)
        self.canvas_compare.get_tk_widget().pack(fill="both", expand=True)

    # ── Architecture Tab ──────────────────────────────────────────────
    def _build_tab_arch(self):
        tab = self.tab_arch
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        ttk.Button(ctrl, text="🧠 Show Network Summary", command=self._show_arch_summary).pack(side="left",padx=4)
        ttk.Button(ctrl, text="📈 Draw Network Graph",   command=self._draw_network_graph).pack(side="left",padx=4)
        ttk.Button(ctrl, text="💾 Export Weights CSV",   command=self._export_weights).pack(side="left",padx=4)

        self.arch_text = tk.Text(tab, font=("Courier New", 10), state="disabled", wrap="word")
        self.arch_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)

        self.fig_arch = plt.figure(figsize=(8, 5))
        arch_plot_fr = ttk.Frame(tab)
        arch_plot_fr.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
        self.canvas_arch = FigureCanvasTkAgg(self.fig_arch, master=arch_plot_fr)
        self.canvas_arch.get_tk_widget().pack(fill="both", expand=True)

    # ── About Tab ─────────────────────────────────────────────────────
    def _build_tab_about(self):
        tab = self.tab_about
        about_text = (
            "MIMO Beamforming Optimizer  v2.0\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Deep Reinforcement Learning for Adaptive Beamforming\n\n"
            "Algorithms:  DQN · Double DQN · Dueling DQN\n"
            "Channels:    Rayleigh · Rician · Saleh-Valenzuela · Custom\n"
            "Codebooks:   DFT · Random · Zadoff-Chu\n\n"
            "Features:\n"
            "  • Live training metrics & animated beam pattern\n"
            "  • BER / Capacity / SNR sweep analysis\n"
            "  • Multi-session comparison dashboard\n"
            "  • Optuna hyperparameter optimization\n"
            "  • Batch experiment runner\n"
            "  • Export: PDF, CSV, JSON, HTML (Plotly)\n"
            "  • Weight export & network graph\n"
            "  • 4 UI themes (Dark, Light, Synthwave, Ocean)\n\n"
            "Libraries: TensorFlow · NumPy · Matplotlib · Optuna · Pandas\n"
            "           fpdf · Plotly\n"
        )
        tk.Label(tab, text=about_text, font=("Courier New", 10),
                 justify="left", anchor="nw").pack(padx=20, pady=20, anchor="nw")

    # ─────────────────────────────────────────────────────────────────
    # THEME ENGINE
    # ─────────────────────────────────────────────────────────────────
    def _configure_styles(self):
        T = self.T
        self.style.configure("TFrame",    background=T["bg2"])
        self.style.configure("TLabel",    background=T["bg2"], foreground=T["fg"],
                             font=("Segoe UI", 9))
        self.style.configure("TButton",   background=T["bg3"], foreground=T["fg"],
                             font=("Segoe UI", 9), padding=5)
        self.style.map("TButton",
                       background=[("active", T["accent"])],
                       foreground=[("active", T["bg"])])
        self.style.configure("TEntry",    fieldbackground=T["bg3"],
                             foreground=T["fg"], insertcolor=T["fg"])
        self.style.configure("TCombobox", fieldbackground=T["bg3"],
                             foreground=T["fg"], selectbackground=T["accent"])
        self.style.configure("TLabelframe",       background=T["bg2"],
                             foreground=T["fg2"])
        self.style.configure("TLabelframe.Label", background=T["bg2"],
                             foreground=T["accent"], font=("Segoe UI", 9, "bold"))
        self.style.configure("TNotebook",         background=T["bg"])
        self.style.configure("TNotebook.Tab",     background=T["bg3"],
                             foreground=T["fg2"], padding=(10, 4))
        self.style.map("TNotebook.Tab",
                       background=[("selected", T["bg2"])],
                       foreground=[("selected", T["accent"])])
        self.style.configure("TProgressbar", troughcolor=T["bg3"],
                             background=T["accent"])
        self.style.configure("TCheckbutton", background=T["bg2"], foreground=T["fg"])
        self.style.configure("Invalid.TEntry", fieldbackground="#c0392b",
                             foreground="white")
        self.style.configure("Treeview", background=T["bg2"], foreground=T["fg"],
                             fieldbackground=T["bg2"], rowheight=22)
        self.style.configure("Treeview.Heading", background=T["bg3"],
                             foreground=T["accent"])

    def _apply_full_theme(self):
        T = self.T
        self._configure_styles()
        self.root.configure(bg=T["bg"])
        self.sidebar.configure(bg=T["bg2"])
        # matplotlib figures
        for fig in [self.fig, self.fig2, self.fig3]:
            fig.set_facecolor(T["plot_bg"])
        self._apply_plot_style()
        try:
            self.canvas.draw()
        except Exception:
            pass
        # log console
        try:
            self.log_text.configure(bg=T["bg3"], fg=T["fg"],
                                    insertbackground=T["fg"])
        except Exception:
            pass

    def _on_theme_change(self, event=None):
        self.current_theme = self.theme_combo.get()
        self.T = THEMES[self.current_theme]
        self._apply_full_theme()
        self._log(f"Theme changed to {self.current_theme}")

    def _apply_plot_style(self):
        T = self.T
        for fig in [self.fig, self.fig2, self.fig3]:
            fig.set_facecolor(T["plot_bg"])
            for ax in fig.axes:
                ax.set_facecolor(T["plot_bg"])
                ax.xaxis.label.set_color(T["text"])
                ax.yaxis.label.set_color(T["text"])
                ax.title.set_color(T["text"])
                ax.tick_params(colors=T["text"])
                for spine in ax.spines.values():
                    spine.set_edgecolor(T["border"])

    # ─────────────────────────────────────────────────────────────────
    # PARAMETER HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _load_preset(self, event=None):
        preset = self.preset_combo.get()
        if preset in PRESETS:
            for key, value in PRESETS[preset].items():
                if key in self.entries:
                    self.entries[key].delete(0, tk.END)
                    self.entries[key].insert(0, str(value))
            self._log(f"Preset '{preset}' loaded.")

    def _reset_defaults(self):
        for key, param in self.params.items():
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, str(param["default"]))
        self._log("Parameters reset to defaults.")

    def _validate_params(self):
        try:
            result = {}
            for key, param in self.params.items():
                entry = self.entries[key]
                raw   = entry.get().strip()
                val   = param["type"](raw)
                if not (param["min"] <= val <= param["max"]):
                    entry.configure(style="Invalid.TEntry")
                    raise ValueError(f"{param['label']} must be in [{param['min']}, {param['max']}]")
                entry.configure(style="TEntry")
                result[key] = val
            if result["epsilon_min"] > result["epsilon"]:
                self.entries["epsilon_min"].configure(style="Invalid.TEntry")
                raise ValueError("Min ε must be ≤ ε")
            result["algo"]         = self.algo_combo.get()
            result["channel_model"]= self.channel_combo.get()
            result["codebook_type"]= self.codebook_combo.get()
            result["hidden_units"] = int(float(self.hidden_units_entry.get()))
            result["dropout_rate"] = float(self.dropout_entry.get())
            result["mobility"]     = float(self.mobility_entry.get())
            result["bandwidth_mhz"]= float(self.bandwidth_entry.get())
            result["use_batch_norm"]= self.batch_norm_var.get()
            return result
        except Exception as ex:
            messagebox.showerror("Invalid Parameters", str(ex))
            return None

    def _apply_params(self):
        params = self._validate_params()
        if params and self.agent:
            with self.param_lock:
                self.agent.gamma         = params["gamma"]
                self.agent.epsilon       = params["epsilon"]
                self.agent.epsilon_min   = params["epsilon_min"]
                self.agent.epsilon_decay = params["epsilon_decay"]
                self.agent.learning_rate = params["learning_rate"]
            self._log("Live params updated.")

    def _upload_channel(self):
        fn = filedialog.askopenfilename(
            filetypes=[("NumPy","*.npy"),("CSV","*.csv"),("All","*.*")])
        if not fn:
            return
        try:
            if fn.endswith(".npy"):
                ch = np.load(fn)
            else:
                ch = pd.read_csv(fn).iloc[0].values
            if self.env and len(ch) != self.env.num_antennas:
                raise ValueError(f"Channel length {len(ch)} ≠ num_antennas {self.env.num_antennas}")
            if self.env:
                self.env.channel = ch.astype(complex)
                self.channel_combo.set("Custom")
            self._log(f"Custom channel loaded ({len(ch)} elements)")
        except Exception as ex:
            messagebox.showerror("Channel Load Error", str(ex))

    # ─────────────────────────────────────────────────────────────────
    # TRAINING CONTROL
    # ─────────────────────────────────────────────────────────────────
    def _start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Training", "Already running!")
            return
        params = self._validate_params()
        if not params:
            return
        self.stop_training  = False
        self.pause_training = False
        self.rewards  = {}; self.rates   = {}
        self.losses   = {}; self.q_values= {}
        self.ber_history     = {}
        self.capacity_history= {}
        self.start_time = time.time()
        self.training_step_count = 0
        self.train_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.stop_button.config(state="normal")
        self.resume_button.config(state="disabled")
        self._log("Training started.")
        self.training_thread = threading.Thread(
            target=self._train_loop, args=(params,), daemon=True)
        self.training_thread.start()

    def _pause_training(self):
        self.pause_training = not self.pause_training
        txt = "⏸ Pause" if not self.pause_training else "▶ Resume"
        self.pause_button.config(text=txt)
        self._log("Paused." if self.pause_training else "Resumed.")

    def _stop_training(self):
        self.stop_training  = True
        self.pause_training = False
        self.status_label.config(text="Stopping…")
        self._log("Stop requested.")

    def _resume_training(self):
        if not self.training_state:
            messagebox.showerror("Error", "No saved state.")
            return
        self.stop_training  = False
        self.pause_training = False
        self.train_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.stop_button.config(state="normal")
        self.resume_button.config(state="disabled")
        self._log("Resuming…")
        self.training_thread = threading.Thread(
            target=self._train_loop,
            args=(self.training_state["params"],
                  self.training_state["episode"]),
            daemon=True)
        self.training_thread.start()

    # ─────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────
    def _train_loop(self, params, resume_ep=0):
        try:
            self.env = MIMOEnvironment(
                num_antennas=params["num_antennas"],
                num_beams=params["num_beams"],
                snr_db=params["snr_db"],
                channel_model=params["channel_model"],
                codebook_type=params["codebook_type"],
                mobility=params["mobility"],
                bandwidth_mhz=params["bandwidth_mhz"],
            )
            algo = params["algo"]

            if self.training_state and resume_ep > 0:
                self.agent            = self.training_state["agent"]
                self.rewards          = self.training_state["rewards"]
                self.rates            = self.training_state["rates"]
                self.losses           = self.training_state["losses"]
                self.q_values         = self.training_state["q_values"]
                self.ber_history      = self.training_state.get("ber_history", {})
                self.capacity_history = self.training_state.get("capacity_history", {})
            else:
                self.agent = DQNAgent(
                    state_size=self.env.state_size,
                    action_size=self.env.action_size,
                    gamma=params["gamma"],
                    epsilon=params["epsilon"],
                    epsilon_min=params["epsilon_min"],
                    epsilon_decay=params["epsilon_decay"],
                    learning_rate=params["learning_rate"],
                    algo=algo,
                    hidden_units=params["hidden_units"],
                    dropout_rate=params["dropout_rate"],
                    use_batch_norm=params["use_batch_norm"],
                )

            target_update_freq = 10  # episodes
            last_ber = 0.0; last_cap = 0.0; last_loss = 0.0

            for episode in range(resume_ep, params["episodes"]):
                if self.stop_training:
                    break
                while self.pause_training:
                    if self.stop_training:
                        break
                    time.sleep(0.1)

                state        = self.env.reset()
                total_reward = 0.0
                ep_ber       = []
                ep_cap       = []
                info         = {"rate": 0.0, "ber": 0.0, "capacity_mbps": 0.0,
                                "beam_vector": None, "signal_power": 0.0, "beam_idx": 0}

                for _step in range(100):
                    if self.stop_training or self.pause_training:
                        break
                    action = self.agent.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    state        = next_state
                    total_reward += reward
                    if info.get("beam_vector") is not None:
                        self.current_beam = info["beam_vector"]
                    self.current_channel = self.env.channel.copy()
                    ep_ber.append(info["ber"])
                    ep_cap.append(info["capacity_mbps"])

                    if len(self.agent.memory) > params["batch_size"]:
                        last_loss = self.agent.replay(params["batch_size"])
                        self.training_step_count += 1
                        mean_q = self.agent.q_history[-1] if self.agent.q_history else 0.0
                        self.q_values.setdefault(algo, []).append(mean_q)
                        self.losses.setdefault(algo, []).append(last_loss)

                    if done:
                        break

                if episode % target_update_freq == 0:
                    self.agent.update_target_model()

                self.rewards.setdefault(algo, []).append(total_reward)
                self.rates.setdefault(algo, []).append(info["rate"])
                last_ber = float(np.mean(ep_ber)) if ep_ber else 0.0
                last_cap = float(np.mean(ep_cap)) if ep_cap else 0.0
                self.ber_history.setdefault(algo, []).append(last_ber)
                self.capacity_history.setdefault(algo, []).append(last_cap)

                # UI update
                elapsed      = time.time() - self.start_time
                rew_list     = self.rewards[algo]
                avg_rew      = float(np.mean(rew_list[-100:]))
                max_rate     = float(max(self.rates[algo]))
                mem_size     = len(self.agent.memory)

                self.root.after_idle(
                    self._update_status,
                    episode, params["episodes"],
                    total_reward, info["rate"],
                    self.agent.epsilon, avg_rew, max_rate,
                    last_loss, last_ber, last_cap, mem_size, elapsed
                )

                if episode % 5 == 0:
                    self.root.after_idle(self._update_plots)

                self.training_state = {
                    "agent":    self.agent,
                    "rewards":  self.rewards,
                    "rates":    self.rates,
                    "losses":   self.losses,
                    "q_values": self.q_values,
                    "ber_history":      self.ber_history,
                    "capacity_history": self.capacity_history,
                    "params":   params,
                    "episode":  episode + 1,
                }

            self.root.after_idle(self._training_complete)
        except Exception as ex:
            tb = traceback.format_exc()
            self._log(f"ERROR: {ex}\n{tb}")
            self.root.after_idle(
                lambda: messagebox.showerror("Training Error", str(ex)))
            self.root.after_idle(self._training_complete)

    def _training_complete(self):
        self.train_button.config(state="normal")
        self.pause_button.config(state="disabled", text="⏸ Pause")
        self.stop_button.config(state="disabled")
        self.resume_button.config(
            state="normal" if self.training_state else "disabled")
        self.progress["value"] = 0
        self.status_label.config(text="Training Complete ✓")
        self._log("Training finished.")
        self._update_plots()

    # ─────────────────────────────────────────────────────────────────
    # ANIMATION  (beam pattern on polar axes)
    # ─────────────────────────────────────────────────────────────────
    def _animate(self, frame):
        T = self.T
        # Only update beam-pattern axis if it exists
        for ax in self.fig.axes:
            if hasattr(ax, '_is_beam_polar') and ax._is_beam_polar:
                ax.clear()
                if self.current_beam is not None and self.env is not None:
                    theta   = np.linspace(0, 2 * np.pi, 360)
                    pattern = np.abs(np.sum(
                        self.current_beam[:, None] *
                        np.exp(1j * np.arange(self.env.num_antennas)[:, None] * theta),
                        axis=0))
                    pattern /= (pattern.max() + 1e-10)
                    ax.plot(theta, pattern, color=T["accent"], linewidth=1.5)
                    ax.fill(theta, pattern, alpha=0.2, color=T["accent"])
                    ax.set_title("Beam Pattern", color=T["text"], fontsize=9)
                    ax.set_facecolor(T["plot_bg"])
                    ax.tick_params(colors=T["text"], labelsize=7)
                else:
                    ax.set_title("Beam Pattern", color=T["fg2"], fontsize=9)
                    try:
                        ax.text(np.pi / 2, 1.0, "Waiting…", ha='center',
                                va='center', color=T["fg2"])
                    except Exception:
                        pass
                break

    # ─────────────────────────────────────────────────────────────────
    # PLOT UPDATE
    # ─────────────────────────────────────────────────────────────────
    def _update_plots(self):
        if not hasattr(self, 'fig'):
            return
        T    = self.T
        self.fig.clear()

        # Determine active plots
        active = [k for k, v in self.plot_config.items() if v]
        n      = len(active)
        if n == 0:
            self.canvas.draw()
            return

        cols   = min(n, 3)
        rows   = math.ceil(n / cols)
        gs     = gridspec.GridSpec(rows, cols, figure=self.fig,
                                   hspace=0.45, wspace=0.35)
        idx    = 0

        colors = [T["accent"], T["accent2"], T["warn"], T["err"]]

        def _smooth(data):
            w = max(1, self.smoothing_window)
            if len(data) >= w:
                kernel = np.ones(w) / w
                return np.convolve(data, kernel, mode='valid'), w - 1
            return np.array(data), 0

        for plot_key in active:
            r, c = divmod(idx, cols)
            is_polar = (plot_key == "beam_pattern")
            ax = self.fig.add_subplot(gs[r, c], polar=is_polar)
            if is_polar:
                ax._is_beam_polar = True
            ax.set_facecolor(T["plot_bg"])

            if plot_key == "rewards":
                for i, (algo, data) in enumerate(self.rewards.items()):
                    ax.plot(data, alpha=0.4, color=colors[i % len(colors)],
                            linewidth=0.8, label=algo)
                    sm, off = _smooth(data)
                    if len(sm) > 1:
                        ax.plot(range(off, off + len(sm)), sm,
                                color=colors[i % len(colors)],
                                linewidth=1.8, label=f"{algo} (smooth)")
                ax.set_title("Rewards", color=T["text"], fontsize=9)
                ax.set_xlabel("Episode", color=T["fg2"], fontsize=8)
                if self.rewards:
                    ax.legend(fontsize=7, facecolor=T["bg3"], labelcolor=T["fg"])

            elif plot_key == "rates":
                for i, (algo, data) in enumerate(self.rates.items()):
                    ax.plot(data, alpha=0.4, color=colors[i % len(colors)],
                            linewidth=0.8, label=algo)
                    sm, off = _smooth(data)
                    if len(sm) > 1:
                        ax.plot(range(off, off + len(sm)), sm,
                                color=colors[i % len(colors)],
                                linewidth=1.8, label=f"{algo} (smooth)")
                ax.set_title("Spectral Rate (bps/Hz)", color=T["text"], fontsize=9)
                ax.set_xlabel("Episode", color=T["fg2"], fontsize=8)
                if self.rates:
                    ax.legend(fontsize=7, facecolor=T["bg3"], labelcolor=T["fg"])

            elif plot_key == "loss_curve":
                for i, (algo, data) in enumerate(self.losses.items()):
                    sm, off = _smooth(data)
                    if len(sm) > 1:
                        safe_sm = np.clip(sm, 1e-12, None)
                        ax.semilogy(range(off, off + len(safe_sm)), safe_sm,
                                    color=colors[i % len(colors)],
                                    linewidth=1.5, label=algo)
                ax.set_title("Training Loss", color=T["text"], fontsize=9)
                ax.set_xlabel("Step", color=T["fg2"], fontsize=8)
                if self.losses:
                    ax.legend(fontsize=7, facecolor=T["bg3"], labelcolor=T["fg"])

            elif plot_key == "q_value":
                for i, (algo, data) in enumerate(self.q_values.items()):
                    sm, off = _smooth(data)
                    if len(sm) > 1:
                        ax.plot(range(off, off + len(sm)), sm,
                                color=colors[i % len(colors)],
                                linewidth=1.5, label=algo)
                ax.set_title("Mean Q-Value", color=T["text"], fontsize=9)
                ax.set_xlabel("Step", color=T["fg2"], fontsize=8)
                if self.q_values:
                    ax.legend(fontsize=7, facecolor=T["bg3"], labelcolor=T["fg"])

            elif plot_key == "beam_pattern":
                # drawn in animation
                ax.set_title("Beam Pattern", color=T["text"], fontsize=9)
                try:
                    ax.text(np.pi / 2, 1.0, "Live…", ha='center',
                            va='center', color=T["fg2"], fontsize=8)
                except Exception:
                    pass

            elif plot_key == "channel_state":
                if self.current_channel is not None and self.env is not None:
                    n_ant = self.env.num_antennas
                    xi    = np.arange(n_ant)
                    ax.bar(xi - 0.2, np.abs(self.current_channel),  0.38,
                           label="Magnitude", alpha=0.85, color=T["accent"])
                    ax.bar(xi + 0.2, np.angle(self.current_channel), 0.38,
                           label="Phase",     alpha=0.85, color=T["accent2"])
                    ax.set_title("Channel State", color=T["text"], fontsize=9)
                    ax.set_xlabel("Antenna", color=T["fg2"], fontsize=8)
                    ax.legend(fontsize=7, facecolor=T["bg3"], labelcolor=T["fg"])
                else:
                    ax.set_title("Channel State", color=T["text"], fontsize=9)
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha='center', va='center', color=T["fg2"])

            elif plot_key == "ber_curve":
                for i, (algo, data) in enumerate(self.ber_history.items()):
                    sm, off = _smooth(data)
                    if len(sm) > 1:
                        safe_sm = np.clip(sm, 1e-12, 1.0)
                        ax.semilogy(range(off, off + len(safe_sm)), safe_sm,
                                    color=colors[i % len(colors)],
                                    linewidth=1.5, label=algo)
                ax.set_title("BER", color=T["text"], fontsize=9)
                ax.set_xlabel("Episode", color=T["fg2"], fontsize=8)
                if self.ber_history:
                    ax.legend(fontsize=7, facecolor=T["bg3"], labelcolor=T["fg"])

            elif plot_key == "capacity":
                for i, (algo, data) in enumerate(self.capacity_history.items()):
                    sm, off = _smooth(data)
                    if len(sm) > 1:
                        ax.plot(range(off, off + len(sm)), sm,
                                color=colors[i % len(colors)],
                                linewidth=1.5, label=algo)
                ax.set_title("Capacity (Mbps)", color=T["text"], fontsize=9)
                ax.set_xlabel("Episode", color=T["fg2"], fontsize=8)
                if self.capacity_history:
                    ax.legend(fontsize=7, facecolor=T["bg3"], labelcolor=T["fg"])

            elif plot_key == "channel_3d":
                ax.remove()
                ax3d = self.fig.add_subplot(gs[r, c], projection='3d')
                ax3d.set_facecolor(T["plot_bg"])
                if self.current_channel is not None and self.env is not None:
                    xa = np.arange(self.env.num_antennas)
                    za = np.abs(self.current_channel)
                    ax3d.bar3d(xa, np.zeros_like(xa), np.zeros_like(za),
                               0.5, 0.5, za, shade=True, color=T["accent"])
                    ax3d.set_title("3-D Channel Magnitude",
                                   color=T["text"], fontsize=9)

            elif plot_key == "beam_heatmap":
                if self.current_beam is not None:
                    hm = np.abs(np.outer(self.current_beam, self.current_beam.conj()))
                    cax = ax.matshow(hm, cmap='viridis', aspect='auto')
                    self.fig.colorbar(cax, ax=ax, shrink=0.8)
                    ax.set_title("Beam Covariance", color=T["text"], fontsize=9)
                    ax.tick_params(colors=T["text"], labelsize=7)

            # Axis styling
            if not is_polar:
                ax.tick_params(colors=T["text"], labelsize=7)
                ax.xaxis.label.set_color(T["fg2"])
                ax.yaxis.label.set_color(T["fg2"])
                ax.title.set_color(T["text"])
                for sp in ax.spines.values():
                    sp.set_edgecolor(T["border"])

            idx += 1

        self.fig.set_facecolor(T["plot_bg"])
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw()

    # ─────────────────────────────────────────────────────────────────
    # STATUS UPDATE
    # ─────────────────────────────────────────────────────────────────
    def _update_status(self, ep, total_ep, reward, rate, eps,
                       avg_rew, max_rate, loss, ber, cap, mem, elapsed):
        pct = (ep + 1) / total_ep * 100
        self.progress["value"] = pct
        self.status_label.config(
            text=f"Ep {ep+1}/{total_ep}  Reward:{reward:.2f}  Rate:{rate:.3f} bps/Hz")
        self.metrics_labels["episode"].config(text=f"Ep: {ep+1}")
        self.metrics_labels["avg_reward"].config(text=f"AvgR: {avg_rew:.2f}")
        self.metrics_labels["max_rate"].config(text=f"MaxR: {max_rate:.3f}")
        self.metrics_labels["epsilon"].config(text=f"ε: {eps:.4f}")
        self.metrics_labels["loss"].config(text=f"Loss: {loss:.5f}")
        self.metrics_labels["ber"].config(text=f"BER: {ber:.2e}")
        self.metrics_labels["capacity"].config(text=f"Cap: {cap:.1f}Mb")
        self.metrics_labels["memory"].config(text=f"Mem: {mem}")
        self.metrics_labels["training_time"].config(text=f"T: {elapsed:.0f}s")

    # ─────────────────────────────────────────────────────────────────
    # PLOT HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _toggle_plot(self, key, value):
        self.plot_config[key] = value
        self._update_plots()

    def _update_smoothing(self, val):
        try:
            self.smoothing_window = max(1, int(float(val)))
            self._update_plots()
        except Exception:
            pass

    def _save_plot(self):
        fn = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("SVG","*.svg"),("PDF","*.pdf")])
        if fn:
            self.fig.savefig(fn, dpi=150, bbox_inches="tight")
            self._log(f"Plot saved → {fn}")

    def _copy_plot_to_clipboard(self):
        try:
            import subprocess
            import tempfile
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf_:
                tf_.write(buf.read())
                tmp = tf_.name
            if os.name == "nt":
                subprocess.run(
                    ["powershell", "-command",
                     f"Add-Type -AssemblyName System.Windows.Forms;"
                     f"[System.Windows.Forms.Clipboard]::SetImage("
                     f"[System.Drawing.Image]::FromFile('{tmp}'))"],
                    check=False)
                self._log("Plot copied to clipboard (Windows).")
            elif sys.platform == "darwin":
                subprocess.run(["osascript", "-e",
                    f'set the clipboard to (read (POSIX file "{tmp}") as JPEG picture)'],
                    check=False)
                self._log("Plot copied to clipboard (macOS).")
            else:
                subprocess.run(["xclip", "-selection", "clipboard",
                                 "-t", "image/png", "-i", tmp],
                    check=False)
                self._log("Plot copied to clipboard (Linux/xclip).")
            os.unlink(tmp)
        except Exception as ex:
            self._log(f"Clipboard copy failed: {ex}")

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS TAB METHODS
    # ─────────────────────────────────────────────────────────────────
    def _refresh_qvalues(self):
        if not self.agent or not self.env or self.current_channel is None:
            messagebox.showinfo("Q-Values", "Train or load a model first.")
            return
        T    = self.T
        qs   = self.agent.get_q_table_sample(self.env)
        self.fig_analysis.clear()
        ax   = self.fig_analysis.add_subplot(111)
        ax.set_facecolor(T["plot_bg"])
        colors_bar = [T["accent"] if i == int(np.argmax(qs)) else T["fg2"]
                      for i in range(len(qs))]
        ax.bar(range(len(qs)), qs, color=colors_bar)
        ax.set_title("Q-Values per Beam (current state)", color=T["text"])
        ax.set_xlabel("Beam Index", color=T["fg2"])
        ax.set_ylabel("Q-Value",    color=T["fg2"])
        ax.tick_params(colors=T["text"])
        self.fig_analysis.set_facecolor(T["plot_bg"])
        self.canvas_analysis.draw()

    def _plot_ber_vs_snr(self):
        if not self.ber_history:
            messagebox.showinfo("BER", "No BER data yet. Run training first.")
            return
        T = self.T
        self.fig_analysis.clear()
        ax = self.fig_analysis.add_subplot(111)
        ax.set_facecolor(T["plot_bg"])
        for algo, data in self.ber_history.items():
            safe_data = np.clip(data, 1e-12, 1.0)
            ax.semilogy(safe_data, label=algo, linewidth=1.5, color=T["accent"])
        ax.set_title("BER per Episode", color=T["text"])
        ax.set_xlabel("Episode",        color=T["fg2"])
        ax.set_ylabel("BER",            color=T["fg2"])
        ax.tick_params(colors=T["text"])
        ax.legend(facecolor=T["bg3"], labelcolor=T["fg"])
        self.fig_analysis.set_facecolor(T["plot_bg"])
        self.canvas_analysis.draw()

    def _plot_capacity_curve(self):
        if not self.capacity_history:
            messagebox.showinfo("Capacity", "No capacity data yet.")
            return
        T = self.T
        self.fig_analysis.clear()
        ax = self.fig_analysis.add_subplot(111)
        ax.set_facecolor(T["plot_bg"])
        for algo, data in self.capacity_history.items():
            ax.plot(data, label=algo, linewidth=1.5, color=T["accent2"])
        ax.set_title("System Capacity (Mbps)", color=T["text"])
        ax.set_xlabel("Episode", color=T["fg2"])
        ax.set_ylabel("Mbps",    color=T["fg2"])
        ax.tick_params(colors=T["text"])
        ax.legend(facecolor=T["bg3"], labelcolor=T["fg"])
        self.fig_analysis.set_facecolor(T["plot_bg"])
        self.canvas_analysis.draw()

    def _plot_codebook_heatmap(self):
        if not self.env:
            messagebox.showinfo("Codebook", "Environment not initialized.")
            return
        T  = self.T
        cb = np.abs(self.env.codebook)     # [num_beams, num_antennas]
        self.fig_analysis.clear()
        ax = self.fig_analysis.add_subplot(111)
        ax.set_facecolor(T["plot_bg"])
        im = ax.imshow(cb, aspect='auto', cmap='viridis', origin='lower')
        self.fig_analysis.colorbar(im, ax=ax, label="Magnitude")
        ax.set_title("Codebook Magnitude Map", color=T["text"])
        ax.set_xlabel("Antenna Index", color=T["fg2"])
        ax.set_ylabel("Beam Index",    color=T["fg2"])
        ax.tick_params(colors=T["text"])
        self.fig_analysis.set_facecolor(T["plot_bg"])
        self.canvas_analysis.draw()

    def _plot_convergence(self):
        if not self.rewards:
            messagebox.showinfo("Convergence", "No training data yet.")
            return
        T = self.T
        self.fig_analysis.clear()
        ax = self.fig_analysis.add_subplot(111)
        ax.set_facecolor(T["plot_bg"])
        for algo, data in self.rewards.items():
            cummax = np.maximum.accumulate(data)
            ax.plot(cummax, label=f"{algo} CumMax", linewidth=1.8,
                    color=T["accent"])
            ax.plot(data, alpha=0.3, color=T["accent"], linewidth=0.8)
        ax.set_title("Cumulative-Max Reward (Convergence)",
                     color=T["text"])
        ax.set_xlabel("Episode", color=T["fg2"])
        ax.tick_params(colors=T["text"])
        ax.legend(facecolor=T["bg3"], labelcolor=T["fg"])
        self.fig_analysis.set_facecolor(T["plot_bg"])
        self.canvas_analysis.draw()

    def _refresh_stats_table(self):
        for row in self.stats_tree.get_children():
            self.stats_tree.delete(row)
        if not self.rewards:
            return
        algo = next(iter(self.rates))
        rew  = self.rewards.get(algo, [])
        rat  = self.rates.get(algo,   [])
        los  = self.losses.get(algo,  [])
        ber  = self.ber_history.get(algo, [])
        cap  = self.capacity_history.get(algo, [])
        rows = [
            ("Algorithm",          algo),
            ("Episodes run",       str(len(rew))),
            ("Mean reward",        f"{np.mean(rew):.4f}"  if rew else "—"),
            ("Max reward",         f"{max(rew):.4f}"      if rew else "—"),
            ("Mean rate (bps/Hz)", f"{np.mean(rat):.4f}"  if rat else "—"),
            ("Max rate (bps/Hz)",  f"{max(rat):.4f}"      if rat else "—"),
            ("Final loss",         f"{los[-1]:.6f}"       if los else "—"),
            ("Mean BER",           f"{np.mean(ber):.2e}"  if ber else "—"),
            ("Mean capacity (Mb)", f"{np.mean(cap):.2f}"  if cap else "—"),
            ("Memory size",        str(len(self.agent.memory)) if self.agent else "—"),
            ("Train steps",        str(self.training_step_count)),
        ]
        for r in rows:
            self.stats_tree.insert("", "end", values=r)

    # ─────────────────────────────────────────────────────────────────
    # SNR SWEEP
    # ─────────────────────────────────────────────────────────────────
    def _run_snr_sweep(self):
        if not self.agent or not self.env:
            messagebox.showerror("Error", "Train or load a model first.")
            return
        try:
            snr_min  = float(self.sweep_min.get())
            snr_max  = float(self.sweep_max.get())
            steps    = int(self.sweep_steps.get())
            eps_pt   = int(self.sweep_eps.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid sweep parameters.")
            return
        snr_range = np.linspace(snr_min, snr_max, steps)

        def _run():
            self._log("SNR sweep started…")
            results = self.env.snr_sweep(snr_range, self.agent, num_episodes=eps_pt)
            self.snr_sweep_result = (snr_range, results)
            self.root.after_idle(self._draw_snr_sweep)

        threading.Thread(target=_run, daemon=True).start()

    def _draw_snr_sweep(self):
        if not self.snr_sweep_result:
            return
        T          = self.T
        snr_range, results = self.snr_sweep_result
        # Also compute theoretical Shannon bound
        theory = np.log2(1 + 10 ** (snr_range / 10.0))
        self.fig2.clear()
        ax = self.fig2.add_subplot(111)
        ax.set_facecolor(T["plot_bg"])
        ax.plot(snr_range, results, "o-", color=T["accent"],
                linewidth=2, markersize=5, label="DRL Agent")
        ax.plot(snr_range, theory,  "--", color=T["warn"],
                linewidth=1.5, label="Shannon Bound")
        ax.set_title("Rate vs SNR Sweep", color=T["text"])
        ax.set_xlabel("SNR (dB)",          color=T["fg2"])
        ax.set_ylabel("Rate (bps/Hz)",     color=T["fg2"])
        ax.tick_params(colors=T["text"])
        ax.legend(facecolor=T["bg3"], labelcolor=T["fg"])
        ax.grid(True, color=T["border"], alpha=0.3)
        self.fig2.set_facecolor(T["plot_bg"])
        self.canvas_sweep.draw()
        self._log("SNR sweep complete.")

    def _export_sweep(self):
        if not self.snr_sweep_result:
            messagebox.showerror("Error", "Run SNR sweep first.")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("PNG","*.png")])
        if not fn:
            return
        snr_range, results = self.snr_sweep_result
        if fn.endswith(".png"):
            self.fig2.savefig(fn, dpi=150, bbox_inches="tight")
        else:
            df = pd.DataFrame({"SNR_dB": snr_range, "Rate_bps_Hz": results})
            df.to_csv(fn, index=False)
        self._log(f"Sweep exported → {fn}")

    # ─────────────────────────────────────────────────────────────────
    # SESSION COMPARE
    # ─────────────────────────────────────────────────────────────────
    def _save_session(self):
        if not self.rewards:
            messagebox.showerror("Error", "No training data to save.")
            return
        label = f"Session {len(self.sessions)+1} — {self.algo_combo.get()}"
        rec = SessionRecord(
            label=label,
            params=self.training_state["params"] if self.training_state else {},
            rewards=copy.deepcopy(self.rewards),
            rates=copy.deepcopy(self.rates),
            losses=copy.deepcopy(self.losses),
            q_values=copy.deepcopy(self.q_values),
            ber_history=copy.deepcopy(self.ber_history),
            capacity_history=copy.deepcopy(self.capacity_history),
        )
        self.sessions.append(rec)
        s = rec.summary()
        self.session_tree.insert("", "end", values=(
            s["label"], s["algo"], s["episodes"],
            s["avg_rate"], s["max_rate"], s["timestamp"]))
        self._log(f"Session saved: {label}")

    def _clear_sessions(self):
        self.sessions.clear()
        for row in self.session_tree.get_children():
            self.session_tree.delete(row)
        self._log("Sessions cleared.")

    def _compare_sessions(self):
        if len(self.sessions) < 2:
            messagebox.showinfo("Compare", "Save at least 2 sessions first.")
            return
        T = self.T
        self.fig3.clear()
        ax1 = self.fig3.add_subplot(121)
        ax2 = self.fig3.add_subplot(122)
        colors = [T["accent"], T["accent2"], T["warn"], T["err"],
                  T["fg"], T["fg2"]]
        for i, sess in enumerate(self.sessions):
            c = colors[i % len(colors)]
            algo = next(iter(sess.rates), "?")
            rates = sess.rates.get(algo, [])
            if rates:
                ax1.plot(rates, alpha=0.6, color=c, linewidth=1,
                         label=sess.label[:20])
                # Smoothed
                w = 50
                if len(rates) > w:
                    sm = np.convolve(rates, np.ones(w)/w, mode='valid')
                    ax1.plot(range(w-1, w-1+len(sm)), sm,
                             color=c, linewidth=2)
            ber = sess.ber_history.get(algo, [])
            if ber:
                safe_ber = np.clip(ber, 1e-12, 1.0)
                ax2.semilogy(safe_ber, alpha=0.6, color=c, linewidth=1,
                             label=sess.label[:20])
        for ax, title, ylabel in [
            (ax1, "Spectral Rate Comparison", "bps/Hz"),
            (ax2, "BER Comparison",           "BER"),
        ]:
            ax.set_facecolor(T["plot_bg"])
            ax.set_title(title, color=T["text"])
            ax.set_xlabel("Episode", color=T["fg2"])
            ax.set_ylabel(ylabel,    color=T["fg2"])
            ax.tick_params(colors=T["text"])
            ax.legend(fontsize=7, facecolor=T["bg3"], labelcolor=T["fg"])
            for sp in ax.spines.values():
                sp.set_edgecolor(T["border"])
        self.fig3.set_facecolor(T["plot_bg"])
        try:
            self.fig3.tight_layout()
        except Exception:
            pass
        self.canvas_compare.draw()

    def _export_comparison(self):
        if not self.sessions:
            messagebox.showerror("Error", "No sessions to export.")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("PNG","*.png"),("HTML","*.html")])
        if not fn:
            return
        if fn.endswith(".csv"):
            rows = []
            for sess in self.sessions:
                algo = next(iter(sess.rates), "?")
                for ep, (r, rt) in enumerate(zip(
                        sess.rewards.get(algo,[]),
                        sess.rates.get(algo,[]))):
                    rows.append({"session":sess.label,"episode":ep,
                                 "reward":r,"rate":rt})
            pd.DataFrame(rows).to_csv(fn, index=False)
        elif fn.endswith(".png"):
            self.fig3.savefig(fn, dpi=150, bbox_inches="tight")
        elif fn.endswith(".html"):
            fig = go.Figure()
            for sess in self.sessions:
                algo = next(iter(sess.rates), "?")
                rates = sess.rates.get(algo, [])
                fig.add_scatter(x=list(range(len(rates))), y=rates,
                                name=sess.label)
            fig.update_layout(title="Session Rate Comparison",
                              xaxis_title="Episode",
                              yaxis_title="Rate (bps/Hz)")
            fig.write_html(fn)
        self._log(f"Comparison exported → {fn}")

    # ─────────────────────────────────────────────────────────────────
    # ARCHITECTURE TAB
    # ─────────────────────────────────────────────────────────────────
    def _show_arch_summary(self):
        if not self.agent:
            messagebox.showinfo("Architecture", "No model loaded.")
            return
        buf = io.StringIO()
        self.agent.model.summary(print_fn=lambda x: buf.write(x + "\n"))
        summary = buf.getvalue()
        agent_d = self.agent.get_summary_dict()
        info = "\n".join(f"  {k}: {v}" for k, v in agent_d.items())
        full = f"=== AGENT CONFIG ===\n{info}\n\n=== KERAS SUMMARY ===\n{summary}"

        self.arch_text.configure(state="normal")
        self.arch_text.delete("1.0", "end")
        self.arch_text.insert("end", full)
        self.arch_text.configure(state="disabled")

    def _draw_network_graph(self):
        if not self.agent:
            messagebox.showinfo("Network Graph", "No model loaded.")
            return
        T  = self.T
        self.fig_arch.clear()
        ax = self.fig_arch.add_subplot(111)
        ax.set_facecolor(T["plot_bg"])
        ax.axis("off")

        layers = self.agent.model.layers
        n_lay  = len(layers)
        x_pos  = np.linspace(0.05, 0.95, n_lay)

        for i, layer in enumerate(layers):
            cfg     = layer.get_config()
            # Robustly extract neuron count across Keras versions
            if "units" in cfg:
                n_units = cfg["units"]
            elif hasattr(layer, "output_shape"):
                try:
                    shape = layer.output_shape
                    n_units = shape[-1] if isinstance(shape, (list, tuple)) else 1
                except Exception:
                    n_units = 1
            else:
                n_units = 1
            n_units = max(1, min(n_units, 20))
            y_pos  = np.linspace(0.1, 0.9, n_units)
            for y in y_pos:
                circ = plt.Circle((x_pos[i], y), 0.018,
                                  color=T["accent"], zorder=3)
                ax.add_patch(circ)
            # Connections to next layer
            if i < n_lay - 1:
                cfg_next = layers[i+1].get_config()
                n_next   = cfg_next.get("units", None)
                if n_next is None:
                    try:
                        shape  = layers[i+1].output_shape
                        n_next = shape[-1] if isinstance(shape, (list, tuple)) else 1
                    except Exception:
                        n_next = 1
                n_next     = max(1, min(n_next, 20))
                y_next_pos = np.linspace(0.1, 0.9, n_next)
                for y1 in y_pos:
                    for y2 in y_next_pos:
                        ax.plot([x_pos[i], x_pos[i+1]], [y1, y2],
                                color=T["border"], linewidth=0.4,
                                alpha=0.5, zorder=1)
            # Layer label
            ax.text(x_pos[i], 0.03, layer.__class__.__name__,
                    ha='center', va='bottom', fontsize=7,
                    color=T["fg2"], rotation=30)

        ax.set_title("Network Architecture", color=T["text"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        self.fig_arch.set_facecolor(T["plot_bg"])
        self.canvas_arch.draw()

    def _export_weights(self):
        if not self.agent:
            messagebox.showerror("Error", "No model loaded.")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if fn:
            self.agent.export_weights_csv(fn)
            self._log(f"Weights exported → {fn}")

    # ─────────────────────────────────────────────────────────────────
    # MODEL SAVE / LOAD / TEST
    # ─────────────────────────────────────────────────────────────────
    def _save_model(self):
        if not self.agent:
            messagebox.showerror("Error", "No model to save.")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("H5","*.h5"),("All","*.*")])
        if fn:
            self.agent.save(fn)
            self._log(f"Model saved → {fn}")

    def _load_model(self):
        try:
            params = self._validate_params()
            if not params:
                return
            fn = filedialog.askopenfilename(filetypes=[("H5","*.h5"),("All","*.*")])
            if not fn:
                return
            self.env = MIMOEnvironment(
                params["num_antennas"], params["num_beams"],
                params["snr_db"], params["channel_model"],
                params["codebook_type"],
                mobility=params["mobility"],
                bandwidth_mhz=params["bandwidth_mhz"])
            self.agent = DQNAgent(
                self.env.state_size, self.env.action_size,
                algo=params["algo"],
                hidden_units=params["hidden_units"],
                dropout_rate=params["dropout_rate"])
            self.agent.load(fn)
            self._log(f"Model loaded ← {fn}")
            messagebox.showinfo("Success", "Model loaded!")
        except Exception as ex:
            messagebox.showerror("Load Error", str(ex))

    def _test_agent(self):
        if not self.agent or not self.env:
            messagebox.showerror("Error", "No model loaded.")
            return
        orig_eps = self.agent.epsilon
        self.agent.epsilon = 0.0
        n_ep  = 200
        rates = []; bers = []; caps = []
        for _ in range(n_ep):
            state = self.env.reset()
            action = self.agent.act_greedy(state)
            _, _, _, info = self.env.step(action)
            rates.append(info["rate"])
            bers.append(info["ber"])
            caps.append(info["capacity_mbps"])
        self.agent.epsilon = orig_eps
        # Compute optimal
        opt_rates = []
        for _ in range(n_ep):
            self.env.reset()
            _, opt_r = self.env.optimal_beam()
            opt_rates.append(opt_r)
        avg_r = np.mean(rates); opt_r = np.mean(opt_rates)
        gap   = (opt_r - avg_r) / opt_r * 100 if opt_r > 0 else 0
        msg   = (f"Episodes: {n_ep}\n"
                 f"Avg Rate:   {avg_r:.4f} bps/Hz\n"
                 f"Optimal:    {opt_r:.4f} bps/Hz\n"
                 f"Gap:        {gap:.1f}%\n"
                 f"Avg BER:    {np.mean(bers):.2e}\n"
                 f"Avg Cap:    {np.mean(caps):.2f} Mbps")
        messagebox.showinfo("Test Results", msg)
        self._log(f"Test: AvgRate={avg_r:.4f} Opt={opt_r:.4f} Gap={gap:.1f}%")

    # ─────────────────────────────────────────────────────────────────
    # EXPORT REPORT
    # ─────────────────────────────────────────────────────────────────
    def _export_report(self):
        if not self.rewards:
            messagebox.showerror("Error", "No data to export.")
            return
        filetypes = [
            ("PDF","*.pdf"),("CSV","*.csv"),
            ("JSON","*.json"),("HTML","*.html"),
            ("Pickle","*.pkl"),
        ]
        fn = filedialog.asksaveasfilename(filetypes=filetypes,
                                          defaultextension=".csv")
        if not fn:
            return
        ext = os.path.splitext(fn)[1].lower()
        if not ext:
            ext = ".csv"
            fn  = fn + ext
        try:
            if ext == ".pdf":
                self._export_pdf(fn)
            elif ext == ".csv":
                self._export_csv(fn)
            elif ext == ".json":
                self._export_json(fn)
            elif ext == ".html":
                self._export_html(fn)
            elif ext == ".pkl":
                with open(fn, "wb") as f:
                    pickle.dump({
                        "rewards": self.rewards, "rates": self.rates,
                        "losses": self.losses,   "q_values": self.q_values,
                        "ber": self.ber_history, "cap": self.capacity_history,
                    }, f)
            self._log(f"Report exported → {fn}")
        except Exception as ex:
            messagebox.showerror("Export Error", str(ex))

    def _export_pdf(self, fn):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "MIMO Beamforming — Training Report", align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        pdf.ln()
        algo = next(iter(self.rates))
        rat  = self.rates[algo]
        ber  = self.ber_history.get(algo, [])
        cap  = self.capacity_history.get(algo, [])
        for txt in [
            f"Algorithm: {algo}  |  Episodes: {len(rat)}",
            f"Max Rate: {max(rat):.4f} bps/Hz  |  Avg Rate: {np.mean(rat):.4f}",
            f"Avg BER: {np.mean(ber):.2e}" if ber else "BER: N/A",
            f"Avg Capacity: {np.mean(cap):.2f} Mbps" if cap else "Cap: N/A",
        ]:
            pdf.cell(0, 7, txt)
            pdf.ln()
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.fig.savefig(tmp.name, dpi=130, bbox_inches="tight")
        tmp.close()
        pdf.image(tmp.name, x=10, w=180)
        pdf.output(fn)
        os.unlink(tmp.name)

    def _export_csv(self, fn):
        algo = next(iter(self.rewards))
        rew  = self.rewards[algo]
        rat  = self.rates[algo]
        ber  = self.ber_history.get(algo, [None]*len(rew))
        cap  = self.capacity_history.get(algo, [None]*len(rew))
        n    = len(rew)
        df   = pd.DataFrame({
            "episode": range(n),
            "reward":  rew,
            "rate":    rat,
            "ber":     ber[:n],
            "capacity_mbps": cap[:n],
        })
        df.to_csv(fn, index=False)

    def _export_json(self, fn):
        data = {
            "rewards":  {k: [float(x) for x in v] for k, v in self.rewards.items()},
            "rates":    {k: [float(x) for x in v] for k, v in self.rates.items()},
            "losses":   {k: [float(x) for x in v] for k, v in self.losses.items()},
            "q_values": {k: [float(x) for x in v] for k, v in self.q_values.items()},
            "ber":      {k: [float(x) for x in v] for k, v in self.ber_history.items()},
            "capacity": {k: [float(x) for x in v] for k, v in self.capacity_history.items()},
            "agent":    self.agent.get_summary_dict() if self.agent else {},
        }
        with open(fn, "w") as f:
            json.dump(data, f, indent=2)

    def _export_html(self, fn):
        fig = go.Figure()
        for algo, rewards in self.rewards.items():
            fig.add_scatter(x=list(range(len(rewards))), y=rewards,
                            name=f"{algo} Reward", opacity=0.6)
        for algo, rates in self.rates.items():
            fig.add_scatter(x=list(range(len(rates))), y=rates,
                            name=f"{algo} Rate", line=dict(width=2))
        for algo, ber in self.ber_history.items():
            fig.add_scatter(x=list(range(len(ber))), y=ber,
                            name=f"{algo} BER", yaxis="y2", opacity=0.7)
        fig.update_layout(
            title="MIMO Beamforming — Training Report",
            xaxis_title="Episode",
            yaxis_title="Reward / Rate",
            yaxis2=dict(title="BER", overlaying="y", side="right",
                        type="log"),
            template="plotly_dark",
        )
        fig.write_html(fn)

    # ─────────────────────────────────────────────────────────────────
    # OPTUNA OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────
    def _optimize_params(self):
        def objective(trial):
            p = {
                "num_antennas":  trial.suggest_int("num_antennas", 2, 8),
                "num_beams":     trial.suggest_int("num_beams", 8, 32),
                "snr_db":        trial.suggest_float("snr_db", -5, 20),
                "episodes":      80,
                "batch_size":    trial.suggest_int("batch_size", 16, 64),
                "gamma":         trial.suggest_float("gamma", 0.88, 0.999),
                "epsilon":       1.0,
                "epsilon_min":   trial.suggest_float("epsilon_min", 0.005, 0.1),
                "epsilon_decay": trial.suggest_float("epsilon_decay", 0.985, 0.9995),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
                "hidden_units":  trial.suggest_categorical("hidden_units", [32,64,128]),
                "dropout_rate":  trial.suggest_float("dropout_rate", 0.0, 0.3),
                "algo":          self.algo_combo.get(),
                "channel_model": self.channel_combo.get(),
                "codebook_type": self.codebook_combo.get(),
                "mobility":      0.05,
                "bandwidth_mhz": 10.0,
                "use_batch_norm":False,
            }
            env   = MIMOEnvironment(p["num_antennas"], p["num_beams"], p["snr_db"],
                                    p["channel_model"], p["codebook_type"],
                                    p["mobility"], frequency_ghz=2.4,
                                    bandwidth_mhz=p["bandwidth_mhz"])
            agent = DQNAgent(env.state_size, env.action_size,
                             gamma=p["gamma"], epsilon=p["epsilon"],
                             epsilon_min=p["epsilon_min"],
                             epsilon_decay=p["epsilon_decay"],
                             learning_rate=p["learning_rate"],
                             algo=p["algo"],
                             hidden_units=p["hidden_units"],
                             dropout_rate=p["dropout_rate"])
            total_rewards = []
            for _ in range(p["episodes"]):
                state = env.reset()
                ep_r  = 0.0
                for _ in range(50):
                    action = agent.act(state)
                    ns, r, done, _ = env.step(action)
                    agent.remember(state, action, r, ns, done)
                    state = ns; ep_r += r
                    if len(agent.memory) > p["batch_size"]:
                        agent.replay(p["batch_size"])
                    if done: break
                total_rewards.append(ep_r)
            return float(np.mean(total_rewards[-30:]))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=15,
                       callbacks=[lambda s, t: self._log(
                           f"Optuna trial {t.number}: {t.value:.3f}")])
        best = study.best_params
        for k, v in best.items():
            if k in self.entries:
                self.entries[k].delete(0, tk.END)
                self.entries[k].insert(0, str(round(v, 6)))
        self._log(f"Optuna best: {best}  val={study.best_value:.3f}")
        messagebox.showinfo("Optuna Done",
                            f"Best trial value: {study.best_value:.4f}\n"
                            f"Params updated in sidebar.")

    # ─────────────────────────────────────────────────────────────────
    # BATCH MODE
    # ─────────────────────────────────────────────────────────────────
    def _batch_mode(self):
        params = self._validate_params()
        if not params:
            return
        self.batch_configs.append(params)
        n = len(self.batch_configs)
        messagebox.showinfo("Batch", f"Config #{n} queued.\n"
                            f"Total queued: {n}\n"
                            "Use 'Train' to run the next config,\n"
                            "or implement auto-batch in _run_batch().")
        self._log(f"Batch config #{n} added: algo={params['algo']}")

    # ─────────────────────────────────────────────────────────────────
    # LOG HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _log(self, message):
        ts  = datetime.now().strftime("%H:%M:%S")
        msg = f"[{ts}] {message}"
        self.log_messages.append(msg)
        try:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        except Exception:
            pass

    def _clear_log(self):
        self.log_messages.clear()
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _save_log(self):
        fn = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text","*.txt")])
        if fn:
            with open(fn, "w", encoding="utf-8") as f:
                f.write("\n".join(self.log_messages))
            self._log(f"Log saved → {fn}")

    # ─────────────────────────────────────────────────────────────────
    # TOOLTIP
    # ─────────────────────────────────────────────────────────────────
    def _show_tooltip(self, event, text):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{event.x_root+12}+{event.y_root+12}")
        T = self.T
        tk.Label(self.tooltip, text=text,
                 background=T["bg3"], foreground=T["fg"],
                 relief="solid", borderwidth=1,
                 font=("Segoe UI", 8), padx=6, pady=3).pack()

    def _hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = MIMOBeamformingGUI(root)
    root.mainloop()



