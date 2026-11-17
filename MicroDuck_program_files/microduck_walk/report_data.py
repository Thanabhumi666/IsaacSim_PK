"""Extra data and figures for the full report (no simulator needed).

  python report_data.py

Writes into report_figures/:
  training/    fig_train_curves.png, fig_train_terms.png, training_summary.txt, tb_tags.txt,
               fig_runs_snapshot.png, fig_gait_sequence.png, fig_early_vs_final.png
  kinematics/  fig_kinematic_diagram.png, fig_velocity_validation.png, velocity_validation.txt,
               joint_load.csv, joint_load.txt, kinematics_result.txt, contacts_result.txt
  gait/        compare_gait_diagram.png, compare_summary.txt, gait_run4_0p15_*.png
Each part runs on its own; a failure is printed and the next part still runs.
"""

import glob
import math
import os
import re
import shutil
import traceback
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs", "rsl_rl", "microduck_flat")
OUT = os.path.join(HERE, "report_figures")
for sub in ("training", "kinematics", "gait"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

RUNS = [  # label, run-dir pattern, console log
    ("Run 1 (humanoid defaults)", "*_16-48-38", "train_log.txt"),
    ("Run 2 (stdfix)", "*_stdfix", "train2_log.txt"),
    ("Run 3 (bipedfix)", "*_bipedfix", "train3_log.txt"),
    ("Run 4 (upright, walks)", "*_upright", "train4_log.txt"),
]
COLORS = ["#7f7f7f", "#ff7f0e", "#d62728", "#1f77b4"]


def run_dir(pattern):
    c = sorted(glob.glob(os.path.join(LOGS, pattern)))
    return c[0] if c else None


def part(name):
    def deco(fn):
        def wrapped():
            print(f"--- {name}", flush=True)
            try:
                fn()
                print(f"--- {name}: ok", flush=True)
            except Exception:  # noqa: BLE001
                print(f"--- {name}: FAILED", flush=True)
                traceback.print_exc()
        return wrapped
    return deco


def smooth(y, n=25):
    if len(y) < n:
        return y
    k = np.ones(n) / n
    return np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


# ======================================================================== A. training curves
def read_text(p):
    b = open(p, "rb").read()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16", errors="ignore")
    if b[:3] == b"\xef\xbb\xbf":
        return b[3:].decode("utf-8", errors="ignore")
    if b.count(b"\x00") > len(b) // 4:
        return b.decode("utf-16-le", errors="ignore")
    return b.decode("utf-8", errors="ignore")


def load_tb(d):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        data[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]), np.array([e.wall_time for e in ev]))
    return data


def load_console(p):
    """Fallback: parse rsl_rl's console table (Learning iteration / Mean reward / Mean episode length)."""
    t = read_text(p)
    it, rew, eplen = [], {}, {}
    cur = None
    for line in t.splitlines():
        m = re.search(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", line)
        if m:
            cur = int(m.group(1))
            it.append(cur)
            continue
        m = re.search(r"Mean reward:\s*([-+\d.eE]+)", line)
        if m and cur is not None:
            rew[cur] = float(m.group(1))
        m = re.search(r"Mean episode length:\s*([-+\d.eE]+)", line)
        if m and cur is not None:
            eplen[cur] = float(m.group(1))
    data = {}
    if rew:
        k = np.array(sorted(rew))
        data["Train/mean_reward"] = (k, np.array([rew[i] for i in k]), np.zeros(len(k)))
    if eplen:
        k = np.array(sorted(eplen))
        data["Train/mean_episode_length"] = (k, np.array([eplen[i] for i in k]), np.zeros(len(k)))
    return data


def find_tag(data, *needles):
    for tag in data:
        if all(n.lower() in tag.lower() for n in needles):
            return tag
    return None


TB = {}


@part("training curves")
def training_curves():
    tags_txt, summary = [], []
    for label, pat, logname in RUNS:
        d = run_dir(pat)
        data, src = {}, "none"
        if d:
            try:
                data, src = load_tb(d), "tensorboard"
            except Exception as e:  # noqa: BLE001
                print("tensorboard read failed:", e)
        if not data and os.path.exists(os.path.join(HERE, logname)):
            data, src = load_console(os.path.join(HERE, logname)), "console log"
        TB[label] = data
        tags_txt.append(f"{label}  ({os.path.basename(d) if d else 'missing'}, source: {src})")
        tags_txt += [f"    {t}" for t in sorted(data)]
        row = [label, os.path.basename(d) if d else "-", src]
        tr = find_tag(data, "mean_reward") or find_tag(data, "reward")
        te = find_tag(data, "mean_episode_length")
        tv = find_tag(data, "error_vel_xy")
        if tr:
            s, v, w = data[tr]
            row += [f"iterations logged {int(s.max()) + 1}",
                    f"final mean reward (last 50) {v[-50:].mean():.2f}"]
            if w.any():
                row.append(f"wall time {(w.max() - w.min()) / 3600:.2f} h")
        if te:
            row.append(f"final episode length {data[te][1][-50:].mean():.0f} steps")
        if tv:
            row.append(f"final xy velocity error {data[tv][1][-50:].mean():.3f} m/s")
        for key in ("base_contact", "bad_orientation", "time_out"):
            tt = find_tag(data, "Episode_Termination", key)
            if tt:
                row.append(f"termination {key} {data[tt][1][-50:].mean():.3f}")
        summary.append("  |  ".join(row))
    open(os.path.join(OUT, "training", "tb_tags.txt"), "w").write("\n".join(tags_txt) + "\n")
    open(os.path.join(OUT, "training", "training_summary.txt"), "w").write("\n".join(summary) + "\n")
    print("\n".join(summary))

    panels = [("mean_reward", "mean reward per episode"), ("mean_episode_length", "episode length (steps, max 1000)"),
              ("error_vel_xy", "xy velocity tracking error (m/s)")]
    fig, axs = plt.subplots(len(panels), 1, figsize=(8, 9), sharex=True)
    for ax, (needle, ylabel) in zip(axs, panels):
        for (label, _, _), col in zip(RUNS, COLORS):
            data = TB.get(label, {})
            tag = find_tag(data, needle)
            if tag:
                s, v, _ = data[tag]
                ax.plot(s, v, color=col, alpha=0.2, lw=0.8)
                ax.plot(s, smooth(v), color=col, lw=1.8, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    axs[-1].set_xlabel("training iteration (4096 robots x 24 steps each)")
    axs[0].set_title("Training curves of the four runs (thin = raw, thick = moving average)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training", "fig_train_curves.png"), dpi=150)

    data = TB.get(RUNS[3][0], {})
    fig, axs = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    tags = sorted(t for t in data if t.startswith("Episode_Reward/"))
    cmap = plt.get_cmap("tab20")
    for j, tag in enumerate(tags):
        s, v, _ = data[tag]
        axs[0].plot(s, smooth(v), lw=1.5, color=cmap(j % 20), ls=("-", "--", ":")[j % 3],
                    label=tag.split("/", 1)[1])
    axs[0].set_ylabel("reward term per episode (weighted)")
    axs[0].legend(fontsize=6.5, ncol=3, loc="center right")
    axs[0].set_title("Run 4: reward terms and termination causes during training")
    for tag in sorted(t for t in data if t.startswith("Episode_Termination/")):
        s, v, _ = data[tag]
        axs[1].plot(s, smooth(v), lw=1.6, label=tag.split("/", 1)[1])
    axs[1].set_ylabel("fraction of episodes ended by")
    axs[1].set_xlabel("training iteration")
    axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training", "fig_train_terms.png"), dpi=150)


# ======================================================================== B. frames from videos
def open_video(p):
    try:
        import imageio.v2 as iio

        r = iio.get_reader(p)
        return ("imageio", r, float(r.get_meta_data().get("fps", 50)))
    except Exception as e1:  # noqa: BLE001
        import cv2

        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {p}: {e1}")
        return ("cv2", cap, float(cap.get(cv2.CAP_PROP_FPS) or 50))


def frame_at(v, t):
    kind, r, fps = v
    idx = max(0, int(round(t * fps)))
    if kind == "imageio":
        return np.asarray(r.get_data(idx))
    import cv2

    r.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = r.read()
    if not ok:
        raise RuntimeError(f"no frame {idx}")
    return f[..., ::-1]


def crop(f, x0=0.27, x1=0.73, y0=0.02, y1=0.98):
    h, w = f.shape[:2]
    return f[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def video_of(pattern):
    d = run_dir(pattern)
    if d is None:
        return None
    p = os.path.join(d, "videos", "play", "rl-video-step-0.mp4")
    return p if os.path.exists(p) else None


@part("video frames")
def video_frames():
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.2))
    for ax, (label, pat, _), col in zip(axs, RUNS, COLORS):
        p = video_of(pat)
        ax.axis("off")
        if p is None:
            ax.text(0.5, 0.5, "no video", ha="center", va="center")
        else:
            v = open_video(p)
            ax.imshow(crop(frame_at(v, 6.0)))
        ax.set_title(label, color=col, fontsize=11)
    fig.suptitle("What each run learned (frame at t = 6 s of its play video, command vx = 0.2 m/s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training", "fig_runs_snapshot.png"), dpi=130)

    # runs 1-3 over time: 5 frames each, to show the reward-hacking postures are held, not a single bad frame
    ts = [6.0, 7.5, 9.0, 10.5, 11.8]  # run 1's camera points at the sky before ~5 s
    fig, axs = plt.subplots(3, len(ts), figsize=(2.3 * len(ts), 7.4))
    for row, ((label, pat, _), col) in enumerate(zip(RUNS[:3], COLORS[:3])):
        p = video_of(pat)
        v = open_video(p) if p else None
        for k, t in enumerate(ts):
            ax = axs[row, k]
            ax.axis("off")
            if v is None:
                ax.text(0.5, 0.5, "no video", ha="center", va="center")
                continue
            ax.imshow(crop(frame_at(v, t)))
            ax.set_title(f"{label}\nt = {t:.1f} s" if k == 0 else f"t = {t:.1f} s", fontsize=8,
                         color=col if k == 0 else "black")
    fig.suptitle("Runs 1-3 from t = 6 s to 12 s of play (command vx = 0.2 m/s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training", "fig_old_runs_sequence.png"), dpi=120)

    p = video_of(RUNS[3][1])
    if p:
        v = open_video(p)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(crop(frame_at(v, 6.0), 0.22, 0.78, 0.0, 1.0))
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "training", "fig_robot.png"), dpi=160)
        t0, dt, n = 5.0, 0.04, 7
        fig, axs = plt.subplots(1, n, figsize=(2.2 * n, 3.6))
        for k, ax in enumerate(axs):
            ax.imshow(crop(frame_at(v, t0 + k * dt), 0.3, 0.7, 0.1, 0.95))
            ax.set_title(f"t = {k * dt:.2f} s", fontsize=9)
            ax.axis("off")
        fig.suptitle("Run 4 gait sequence over one step cycle (period about 0.24 s)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "training", "fig_gait_sequence.png"), dpi=130)

        early = os.path.join(HERE, "logs", "upright_early.mp4")
        if os.path.exists(early):
            ve = open_video(early)
            fig, axs = plt.subplots(2, 5, figsize=(12, 5.8))
            for row, (vv, lab) in enumerate(((ve, "early (about iteration 250)"), (v, "final (iteration 2999)"))):
                for k in range(5):
                    ax = axs[row, k]
                    ax.imshow(crop(frame_at(vv, 4.0 + k * 0.8), 0.3, 0.7, 0.05, 0.95))
                    ax.axis("off")
                    ax.set_title(f"{lab}\nt = {4.0 + k * 0.8:.1f} s" if k == 0 else f"t = {4.0 + k * 0.8:.1f} s", fontsize=8)
            fig.suptitle("Run 4: early policy vs. final policy")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "training", "fig_early_vs_final.png"), dpi=130)


# ======================================================================== C. URDF kinematics
URDF = os.path.join(HERE, "assets", "microduck_fixed.urdf")
STANCE = {"left_hip_pitch": -0.3, "left_knee": 0.3, "left_ankle": 0.6,
          "right_hip_pitch": 0.3, "right_knee": -0.3, "right_ankle": -0.6}


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(a, q):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


def tf(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


root = ET.parse(URDF).getroot()
links = {l.get("name"): l for l in root.findall("link")}
joints = {}
for j in root.findall("joint"):
    o = j.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    r = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    ax = j.find("axis")
    joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                 child=j.find("child").get("link"), T=tf(rpy(*r), np.array(xyz)),
                                 axis=np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]))
child_of = {v["child"]: k for k, v in joints.items()}
base = [n for n in links if n not in child_of][0]
REV = [n for n, j in joints.items() if j["type"] in ("revolute", "continuous")]


def fk(q):
    Tl, Tj = {base: np.eye(4)}, {}
    pending = list(joints)
    while pending:
        for name in list(pending):
            j = joints[name]
            if j["parent"] in Tl:
                Tjoint = Tl[j["parent"]] @ j["T"]
                Tj[name] = Tjoint
                R = axis_angle(j["axis"], q.get(name, 0.0)) if j["type"] in ("revolute", "continuous") else np.eye(3)
                Tl[j["child"]] = Tjoint @ tf(R, np.zeros(3))
                pending.remove(name)
    return Tl, Tj


def chain(link):
    js = []
    while link in child_of:
        if joints[child_of[link]]["type"] in ("revolute", "continuous"):
            js.append(child_of[link])
        link = joints[child_of[link]]["parent"]
    return js[::-1]


@part("kinematic diagram")
def kinematic_diagram():
    Tl, Tj = fk(STANCE)
    order = chain("ankle_left") + chain("ankle_right") + [n for n in REV if n not in chain("ankle_left") + chain("ankle_right")]
    num = {n: i + 1 for i, n in enumerate(order)}
    fig, axs = plt.subplots(1, 3, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1.1, 0.9, 0.5]})
    for ax, (i, k, title) in zip(axs, ((1, 2, "front view (y-z plane)"),
                                       (0, 2, "side view (x-z plane)\nright leg hidden behind left"))):
        for name, jd in joints.items():
            a = Tl[jd["parent"]][:3, 3] * 1000
            b = Tj[name][:3, 3] * 1000
            rev = jd["type"] in ("revolute", "continuous")
            ax.plot([a[i], b[i]], [a[k], b[k]], color="0.35" if rev else "0.75", lw=2.2 if rev else 1.0, zorder=1)
        for foot in ("ankle_left", "ankle_right"):
            pass
        for n in order:
            if i == 0 and n.startswith("right"):
                continue  # side view: the right leg sits exactly behind the left one
            p = Tj[n][:3, 3] * 1000
            col = "tab:blue" if n.startswith("left") else ("tab:red" if n.startswith("right") else "tab:green")
            ax.plot(p[i], p[k], "o", ms=9, mfc="white", mec=col, mew=2, zorder=3)
            a = Tj[n][:3, :3] @ (joints[n]["axis"] / np.linalg.norm(joints[n]["axis"]))
            ax.annotate("", xy=(p[i] + 14 * a[i], p[k] + 14 * a[k]), xytext=(p[i], p[k]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2), zorder=2)
            show = (i == 1) or not n.startswith("right")
            if show:
                ax.text(p[i] + 2.5, p[k] + 2.5, str(num[n]), fontsize=8, color=col, zorder=4)
        b0 = Tl[base][:3, 3] * 1000
        ax.plot(b0[i], b0[k], "ks", ms=8, zorder=4)
        ax.text(b0[i] + 3, b0[k] - 6, "base_link", fontsize=8)
        if i == 0:
            ax.set_xlim(-55, 55)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(("y (mm)" if i == 1 else "x forward (mm)"))
        ax.set_ylabel("z up (mm)")
    legend = "\n".join(f"{num[n]:2d}  {n}" for n in order)
    axs[2].axis("off")
    axs[2].text(0.0, 1.0, "revolute joints\n(arrow = rotation axis)\n\n" + legend, transform=axs[2].transAxes,
                va="top", fontsize=9, family="monospace")
    fig.suptitle(f"Kinematic diagram of the Micro Duck at the standing pose: {len(REV)} revolute joints "
                 f"(blue = left leg, red = right leg, green = head/jaw)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kinematics", "fig_kinematic_diagram.png"), dpi=150)


def load_csv(p):
    with open(p) as fh:
        cols = fh.readline().strip().split(",")
    return {c: i for i, c in enumerate(cols)}, np.loadtxt(p, delimiter=",", skiprows=1)


@part("velocity validation")
def velocity_validation():
    col, data = load_csv(os.path.join(HERE, "gait_run4_0p15.csv"))
    t = data[:, col["t"]]
    dt = float(np.median(np.diff(t)))
    names = [n for n in REV if "q_" + n in col]
    Q = {n: data[:, col["q_" + n]] for n in names}
    QD = {n: data[:, col["qd_" + n]] for n in names}
    foot = data[:, [col["footL_x"], col["footL_y"], col["footL_z"]]]
    legL = chain("ankle_left")

    def rel(q):
        Tl, _ = fk(q)
        return Tl["ankle_left"][:3, 3] - Tl["hip_left"][:3, 3]

    N = len(t)
    P = np.zeros((N, 3))
    V = np.zeros((N, 3))
    for k in range(N):
        q = {n: Q[n][k] for n in names}
        p0 = rel(q)
        J = np.zeros((3, len(legL)))
        for c, n in enumerate(legL):
            q2 = dict(q)
            q2[n] += 1e-6
            J[:, c] = (rel(q2) - p0) / 1e-6
        P[k] = p0
        V[k] = J @ np.array([QD[n][k] for n in legL])
    Vfd = np.gradient(foot, dt, axis=0)
    pe = (P - foot) * 1000
    ve = V[2:-2] - Vfd[2:-2]
    lines = ["Velocity analysis check, run 4 at vx = 0.15 m/s, left ankle relative to hip_left (base frame)",
             f"samples {N}, dt {dt:.3f} s",
             "position: forward kinematics from the URDF with the recorded joint angles vs. position recorded in Isaac Sim",
             "  RMS error x/y/z (mm): " + " / ".join(f"{np.sqrt((pe[:, a] ** 2).mean()):.3f}" for a in range(3)),
             "  max |error| (mm): " + f"{np.abs(pe).max():.3f}",
             "velocity: v = J(q) q_dot (URDF Jacobian x recorded joint speeds) vs. finite difference of the recorded position",
             "  RMS of v (m/s) x/y/z: " + " / ".join(f"{np.sqrt((Vfd[:, a] ** 2).mean()):.4f}" for a in range(3)),
             "  RMS difference (m/s) x/y/z: " + " / ".join(f"{np.sqrt((ve[:, a] ** 2).mean()):.4f}" for a in range(3)),
             "  correlation x/y/z: " + " / ".join(f"{np.corrcoef(V[2:-2, a], Vfd[2:-2, a])[0, 1]:.4f}" for a in range(3)),
             "  peak |v| of the ankle relative to the hip (m/s): " + f"{np.abs(V).max():.3f}",
             "(finite differences at 50 Hz smooth the peaks, so a small difference is expected)"]
    open(os.path.join(OUT, "kinematics", "velocity_validation.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    m = t < 1.5
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for c, (a, lab) in enumerate(((0, "x (forward)"), (2, "z (up)"))):
        axs[0, c].plot(t[m], foot[m, a] * 1000, "k", lw=2.5, alpha=0.5, label="Isaac Sim (recorded)")
        axs[0, c].plot(t[m], P[m, a] * 1000, "C1--", lw=1.5, label="forward kinematics (URDF)")
        axs[0, c].set_ylabel(f"ankle {lab} rel. hip (mm)")
        axs[1, c].plot(t[m], Vfd[m, a], "k", lw=2.5, alpha=0.5, label="d/dt of recorded position")
        axs[1, c].plot(t[m], V[m, a], "C0--", lw=1.5, label="J(q) q_dot")
        axs[1, c].set_ylabel(f"ankle velocity {lab} (m/s)")
        axs[1, c].set_xlabel("time (s)")
        for r_ in (0, 1):
            axs[r_, c].grid(alpha=0.3)
            axs[r_, c].legend(fontsize=7)
    fig.suptitle("Position and velocity analysis of the left leg checked against the simulation (run 4, vx 0.15 m/s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kinematics", "fig_velocity_validation.png"), dpi=150)


@part("joint loads")
def joint_loads():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "gait_run4_*.csv"))):
        tag = os.path.basename(p)[10:-4]
        col, data = load_csv(p)
        for n in chain("ankle_left") + chain("ankle_right"):
            if "tau_" + n not in col:
                continue
            tau = data[:, col["tau_" + n]]
            qd = data[:, col["qd_" + n]]
            pw = tau * qd
            rows.append((tag, n, np.abs(tau).max(), np.sqrt((tau ** 2).mean()), np.abs(qd).max(),
                         np.abs(pw).max(), np.clip(pw, 0, None).mean()))
    with open(os.path.join(OUT, "kinematics", "joint_load.csv"), "w") as fh:
        fh.write("speed,joint,peak_abs_tau_Nm,rms_tau_Nm,peak_abs_qd_rad_s,peak_abs_power_W,mean_positive_power_W\n")
        for r_ in rows:
            fh.write(",".join([r_[0], r_[1]] + [f"{x:.4f}" for x in r_[2:]]) + "\n")
    lines = ["Joint loads while walking (run 4). Motor limits in the model: 1.0 Nm, 10 rad/s.",
             f"{'speed':6s} {'joint':18s} {'peak|tau|':>9s} {'rms tau':>8s} {'%limit':>7s} {'peak|qd|':>9s} {'peak|P|':>8s} {'mean P+':>8s}"]
    for r_ in rows:
        lines.append(f"{r_[0]:6s} {r_[1]:18s} {r_[2]:9.3f} {r_[3]:8.3f} {r_[2] * 100:6.1f}% {r_[4]:9.2f} {r_[5]:8.3f} {r_[6]:8.3f}")
    open(os.path.join(OUT, "kinematics", "joint_load.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))


@part("theory figure 2R leg")
def theory_2r():
    l1, l2 = 42.0, 42.0  # sagittal thigh / shin lengths (mm), see limit_position_result.txt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, (th1, th2, title) in zip(axs, ((-100, 60, "bent knee: |det J| large"),
                                           (-90, 0, "toggle position: leg straight, det J = 0"),
                                           (-60, 150, "folded: near the other dead centre"))):
        a1 = math.radians(th1)
        a2 = math.radians(th1 + th2)
        k = (l1 * math.cos(a1), l1 * math.sin(a1))
        f = (k[0] + l2 * math.cos(a2), k[1] + l2 * math.sin(a2))
        ax.plot([0, k[0]], [0, k[1]], lw=6, color="#1f77b4", solid_capstyle="round")
        ax.plot([k[0], f[0]], [k[1], f[1]], lw=6, color="#ff7f0e", solid_capstyle="round")
        for p_, lab in (((0, 0), "hip"), (k, "knee"), (f, "ankle")):
            ax.plot(*p_, "o", ms=11, mfc="white", mec="k", mew=2)
            ax.text(p_[0] + 4, p_[1] + 2, lab, fontsize=9)
        ax.text(k[0] / 2 - 14, k[1] / 2, "l1", fontsize=10, color="#1f77b4")
        ax.text((k[0] + f[0]) / 2 + 4, (k[1] + f[1]) / 2, "l2", fontsize=10, color="#ff7f0e")
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot((l1 + l2) * np.cos(th), (l1 + l2) * np.sin(th), "k:", lw=0.8)
        ax.plot(abs(l1 - l2) * np.cos(th), abs(l1 - l2) * np.sin(th), "k:", lw=0.8)
        ax.set_title(title + f"\nknee angle {th2} deg from straight", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-95, 95)
        ax.set_ylim(-95, 40)
        ax.grid(alpha=0.3)
    fig.suptitle("Planar 2R model of the leg (sagittal plane): outer dotted circle = reach limit l1 + l2 (toggle)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kinematics", "fig_theory_2r.png"), dpi=150)


@part("copy existing results")
def copy_existing():
    for n in ("compare_gait_diagram.png", "compare_summary.txt"):
        if os.path.exists(os.path.join(HERE, n)):
            shutil.copy(os.path.join(HERE, n), os.path.join(OUT, "gait", n))
    for p in glob.glob(os.path.join(HERE, "gait_run4_0p15_*.png")):
        shutil.copy(p, os.path.join(OUT, "gait", os.path.basename(p)))
    for n in ("kinematics_result.txt", "contacts_result.txt"):
        if os.path.exists(os.path.join(HERE, n)):
            shutil.copy(os.path.join(HERE, n), os.path.join(OUT, "kinematics", n))


for fn in (copy_existing, theory_2r, kinematic_diagram, velocity_validation, joint_loads, training_curves, video_frames):
    fn()
for sub in ("training", "kinematics", "gait"):
    print(sub + ":", ", ".join(sorted(os.listdir(os.path.join(OUT, sub)))))
print("REPORTDATA DONE")
