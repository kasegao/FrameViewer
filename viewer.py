import json
import tkinter as tk
import tkinter.filedialog
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from lrupy import LRUCache
from PIL import Image, ImageOps, ImageTk

from paths import *


@dataclass
class FrameViewerSettings:
    last_dir: str = str(path_input)
    cache_size: int = 200

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls(**d)
        if not Path(obj.last_dir).is_dir():
            obj.last_dir = str(path_input)
        return obj

    @classmethod
    def from_json(cls, s: str):
        d = json.loads(s)
        return cls.from_dict(d)


class FrameViewer:
    def __init__(self):
        self.settings = FrameViewerSettings()
        self.load_settings()

        root = tk.Tk()
        root.title("Frame Viewer")
        root.geometry("800x600")
        margin = 5
        tcl_valid_number = root.register(self.validate_frame_no)

        self.default_bg = root.cget("bg")
        self.active_bg = rgb2hex(255, 255, 255)

        # input widgets
        frame_input = tk.Frame(root)
        frame_input.pack(anchor=tk.NW, padx=margin, pady=margin)

        img_open_file = load_image(path_icons / "open_file.png")
        btn_open = tk.Button(frame_input, image=img_open_file, command=self.file_dialog)
        btn_open.bind("<Enter>", self.on_enter)
        btn_open.bind("<Leave>", self.on_leave)
        btn_open.pack(side=tk.LEFT, padx=margin, pady=margin)

        self.file_name = tk.StringVar()
        self.file_name.set("select file")
        label_file_name = tk.Label(
            frame_input, textvariable=self.file_name, font=("Times New Roman", 16)
        )
        label_file_name.pack(side=tk.LEFT, padx=margin, pady=margin)

        # video widgets
        self.image: Optional[tk.PhotoImage] = None
        canvas = tk.Canvas(root, bg="black")
        canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas = canvas

        # controller widgets
        frame_ctrl = tk.Frame(root)
        frame_ctrl.pack(anchor=tk.CENTER, padx=margin, pady=margin, fill=tk.X)

        self.seek_val = tk.IntVar(value=0)
        seek_bar = tk.Scale(
            frame_ctrl,
            variable=self.seek_val,
            orient=tk.HORIZONTAL,
            from_=0,
            to=255,
            resolution=1,
            command=self.on_seek,
        )
        seek_bar.pack(
            side=tk.TOP, anchor=tk.CENTER, padx=margin * 10, pady=margin, fill=tk.X
        )
        self.seek_bar = seek_bar

        frame_ctrl_row1 = tk.Frame(frame_ctrl)
        frame_ctrl_row1.pack(anchor=tk.CENTER, padx=margin, pady=margin)

        img_back = load_image(path_icons / "back.png")
        btn_next = tk.Button(
            frame_ctrl_row1, image=img_back, command=self.previous_frame
        )
        btn_next.pack(side=tk.LEFT, anchor=tk.CENTER, padx=margin, pady=margin)

        entry_frame_no = tk.Entry(
            frame_ctrl_row1,
            width=2,
            font=("Courier New", 16),
            validate="key",
            vcmd=(tcl_valid_number, "%S", "%P"),
        )
        entry_frame_no.pack(side=tk.LEFT, anchor=tk.CENTER, padx=margin, pady=margin)
        entry_frame_no.bind("<Return>", lambda e: self.refresh_frame())
        entry_frame_no.bind("<FocusOut>", lambda e: self.refresh_frame())
        self.entry_frame_no = entry_frame_no

        self.frame_no_str = tk.StringVar()
        self.frame_no_str.set("/ -")
        label_frame_no = tk.Label(
            frame_ctrl_row1, textvariable=self.frame_no_str, font=("Courier New", 16)
        )
        label_frame_no.pack(side=tk.LEFT, padx=margin, pady=margin)

        img_refresh = load_image(path_icons / "refresh.png")
        btn_refresh = tk.Button(
            frame_ctrl_row1, image=img_refresh, command=self.refresh_frame
        )
        btn_refresh.pack(side=tk.LEFT, anchor=tk.CENTER, padx=margin, pady=margin)

        img_forward = load_image(path_icons / "forward.png")
        btn_forward = tk.Button(
            frame_ctrl_row1, image=img_forward, command=self.next_frame
        )
        btn_forward.pack(side=tk.LEFT, anchor=tk.CENTER, padx=margin, pady=margin)

        self.controllers = [
            seek_bar,
            btn_next,
            entry_frame_no,
            btn_refresh,
            btn_forward,
        ]
        self.deactivate_controllers()

        # opencv
        self.video_available = False
        self.video = None
        self.frame: Optional[np.ndarray] = None
        self.frame_count: Optional[int] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.frame_no = tk.IntVar(value=0)
        self.frame_no.trace("w", self.on_frame_no_change)
        self.current_frame_no: int = 0
        self.frame_cache = LRUCache[int, np.ndarray](maxsize=self.settings.cache_size)

        # keyboard
        root.bind("<Left>", lambda e: self.previous_frame())
        root.bind("<Right>", lambda e: self.next_frame())

        # close event
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.root = root
        self.root.mainloop()

    def file_dialog(self):
        filetypes = [("video files", "*.mp4;*.mov;*.avi"), ("all files", "*")]
        file_name = tk.filedialog.askopenfilename(
            filetypes=filetypes, initialdir=self.settings.last_dir
        )
        if len(file_name) == 0:
            self.file_name.set("canceled")
        else:
            self.settings.last_dir = str(Path(file_name).parent)
            self.file_name.set(file_name)
            self.load_video()

    def validate_frame_no(self, diff: str, after: str):
        if not diff.encode("utf-8").isdigit():
            return False
        elif len(after) == 0:
            return True
        elif not after.encode("utf-8").isdigit():
            return False
        else:
            no = int(after)
            return 0 <= no and no < self.frame_count

    def on_seek(self, e: tk.Event):
        seek_val = self.seek_val.get()
        if seek_val == self.frame_no.get():
            return
        self.frame_no.set(seek_val)

    def on_frame_no_change(self, *args):
        new_no = self.frame_no.get()
        frame = self.frame_cache.get_or_else(new_no, self.read_frame)
        if frame is None:
            self.frame_no.set(self.current_frame_no)
            return
        self.seek_val.set(new_no)
        self.entry_frame_no.delete(0, tk.END)
        self.entry_frame_no.insert(0, str(new_no))
        self.current_frame_no = new_no
        self.frame = frame
        self.render_frame()

    def read_frame(self, no: int) -> Optional[np.ndarray]:
        if no != self.current_frame_no + 1:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, no)

        _, frame = self.video.read()
        return frame

    def next_frame(self):
        if not self.video_available:
            return

        if (no := self.frame_no.get() + 1) < self.frame_count:
            self.frame_no.set(no)

    def previous_frame(self):
        if not self.video_available:
            return

        if (no := self.frame_no.get() - 1) >= 0:
            self.frame_no.set(no)

    def refresh_frame(self):
        if not self.video_available:
            return

        frame_no = int(self.entry_frame_no.get())
        if (
            frame_no == self.frame_no.get()
            or frame_no < 0
            or frame_no >= self.frame_count
        ):
            return

        self.frame_no.set(frame_no)

    def load_video(self):
        video = cv2.VideoCapture(self.file_name.get())
        if not video.isOpened():
            print("Error opening video stream or file")
            return

        ret, frame = video.read()
        if not ret:
            print("Error reading video")
            return

        self.video = video
        self.frame = frame
        self.frame_cache.clear()
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frame_no = self.frame_count - 1
        self.frame_no_str.set(f"/ {max_frame_no}")
        self.entry_frame_no.config(width=len(str(max_frame_no)) + 1)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_available = True
        self.render_frame()
        self.activate_controllers()
        self.frame_no.set(0)
        self.seek_val.set(0)
        self.seek_bar.configure(to=max_frame_no)

    def render_frame(self):
        if not self.video_available or self.frame is None:
            return

        image_cv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_cv)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_pil = ImageOps.pad(img_pil, (canvas_width, canvas_height))
        self.image = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(canvas_width / 2, canvas_height / 2, image=self.image)

    def activate_controllers(self):
        for controller in self.controllers:
            controller.configure(state=tk.NORMAL)

    def deactivate_controllers(self):
        for controller in self.controllers:
            controller.configure(state=tk.DISABLED)

    def on_enter(self, e: tk.Event):
        e.widget["background"] = self.active_bg

    def on_leave(self, e: tk.Event):
        e.widget["background"] = self.default_bg

    def save_settings(self):
        with open(path_settings, "w") as f:
            json.dump(self.settings.__dict__, f)

    def load_settings(self):
        if not path_settings.is_file():
            return

        with open(path_settings, "r") as f:
            d = json.load(f)
            self.settings.__dict__.update(d)

    def on_close(self, *args):
        self.save_settings()
        if self.video is not None:
            self.video.release()
        if self.root is not None:
            self.root.destroy()

    def __del__(self):
        self.on_close()


def load_image(path: Path, size: tuple[int, int] = (40, 40)) -> tk.PhotoImage:
    img = Image.open(path)
    img = img.resize(size)
    img = ImageTk.PhotoImage(img)
    return img


def rgb2hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    FrameViewer()
