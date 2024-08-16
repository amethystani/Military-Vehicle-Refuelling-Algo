import os
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
class VideoPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("Simulation Video Player")

        # Define color scheme
        self.BG_COLOR = "#202124"  # Dark background
        self.FG_COLOR = "#e8eaed"  # Light text
        self.ACCENT_COLOR = "#E5E5E2"  # Blue accent
        self.SECONDARY_BG = "#303134"  # Slightly lighter than BG_COLOR
        self.PADDING_COLOR = "#2a2b2e"  # Darker color for padding, between BG_COLOR and SECONDARY_BG

        self.WINDOW_WIDTH = 1280
        self.WINDOW_HEIGHT = 720

        self.master.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.master.configure(bg=self.BG_COLOR)

        self.video_folder = "Videos"
        self.current_video = None
        self.is_playing = False
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 0
        self.is_fullscreen = False

        self.create_widgets()
        self.update_video_list()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.master, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)  # Increased padding

        # Notebook (Tabs)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)  # Increased padding

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Main.TFrame', background=self.BG_COLOR)
        style.configure('TNotebook', background=self.PADDING_COLOR)
        style.configure('TNotebook.Tab', background=self.SECONDARY_BG, foreground=self.FG_COLOR, padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', self.ACCENT_COLOR)],
                  foreground=[('selected', self.BG_COLOR)])
        style.configure('TFrame', background=self.BG_COLOR)
        style.configure('TButton', background=self.SECONDARY_BG, foreground=self.FG_COLOR, borderwidth=1, relief='flat')
        style.map('TButton', background=[('active', self.ACCENT_COLOR)], foreground=[('active', self.BG_COLOR)])
        style.configure('Horizontal.TProgressbar', background=self.ACCENT_COLOR, troughcolor=self.SECONDARY_BG)

        # Videos tab
        videos_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(videos_tab, text='Videos', padding=(10, 10, 10, 10))  # Increased padding
        self.create_video_tab_widgets(videos_tab)

        # Analysis tab
        analysis_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(analysis_tab, text='Analysis', padding=(10, 10, 10, 10))  # Increased padding
        self.create_analysis_tab_widgets(analysis_tab)

        # About tab
        about_tab = ttk.Frame(notebook, style='TFrame')
        notebook.add(about_tab, text='About', padding=(10, 10, 10, 10))  # Increased padding
        self.create_about_tab_widgets(about_tab)

    def create_video_tab_widgets(self, videos_tab):
        # Left frame for video list
        left_frame = ttk.Frame(videos_tab, width=200, style='TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        left_frame.pack_propagate(False)

        # Video listbox
        self.video_listbox = tk.Listbox(left_frame, bg=self.SECONDARY_BG, fg=self.FG_COLOR,
                                        selectbackground=self.ACCENT_COLOR, selectforeground=self.BG_COLOR,
                                        font=("Helvetica", 12), borderwidth=0, highlightthickness=0)
        self.video_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)

        # Progress label
        self.progress_label = tk.Label(left_frame, text="0 / 15 days generated", fg=self.FG_COLOR,
                                       bg=self.SECONDARY_BG, font=("Helvetica", 12))
        self.progress_label.pack(pady=10, padx=5)

        # Right frame for video and controls
        right_frame = ttk.Frame(videos_tab, style='TFrame')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        # Video canvas
        self.canvas = tk.Canvas(right_frame, bg=self.BG_COLOR, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Controls frame
        self.controls_frame = tk.Frame(right_frame, bg=self.BG_COLOR)
        self.controls_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.controls_frame, mode='determinate', length=self.WINDOW_WIDTH - 300)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5, side=tk.TOP)
        self.progress_bar.bind("<Button-1>", self.on_progress_bar_click)

        # Buttons
        button_frame = tk.Frame(self.controls_frame, bg=self.BG_COLOR)
        button_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.TOP)

        button_style = {'bg': self.SECONDARY_BG, 'fg': self.FG_COLOR, 'bd': 1, 'relief': 'flat',
                        'font': ("Helvetica", 10), 'width': 3, 'height': 1}
        self.play_pause_button = tk.Button(button_frame, text="⏵", command=self.toggle_play, **button_style)
        self.play_pause_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = tk.Button(button_frame, text="⏹", command=self.stop, **button_style)
        self.stop_button.pack(side=tk.LEFT, padx=2)

        self.backward_button = tk.Button(button_frame, text="⏪", command=self.backward_10s, **button_style)
        self.backward_button.pack(side=tk.LEFT, padx=2)

        self.forward_button = tk.Button(button_frame, text="⏩", command=self.forward_10s, **button_style)
        self.forward_button.pack(side=tk.LEFT, padx=2)

        self.screenshot_button = tk.Button(button_frame, text="📸", command=self.take_screenshot, **button_style)
        self.screenshot_button.pack(side=tk.LEFT, padx=2)

        self.fullscreen_button = tk.Button(button_frame, text="⛶", command=self.toggle_fullscreen, **button_style)
        self.fullscreen_button.pack(side=tk.LEFT, padx=2)

        # Time label
        self.time_label = tk.Label(button_frame, text="00:00 / 00:00", fg=self.FG_COLOR, bg=self.BG_COLOR,
                                   font=("Helvetica", 10))
        self.time_label.pack(side=tk.RIGHT, padx=5)

    def create_analysis_tab_widgets(self, analysis_tab):
        # Main frame for analysis tab
        main_frame = ttk.Frame(analysis_tab, style='TNotebook')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for file list
        left_frame = ttk.Frame(main_frame, style='TNotebook', width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        left_frame.pack_propagate(False)

        # Right frame for content display
        self.right_frame = ttk.Frame(main_frame, style='TNotebook')
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        # File list
        self.file_listbox = tk.Listbox(left_frame, bg=self.SECONDARY_BG, fg=self.FG_COLOR,
                                       selectbackground=self.ACCENT_COLOR, selectforeground=self.BG_COLOR,
                                       font=("Helvetica", 10))
        self.file_listbox.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_insight_file_select)

        # Populate file list
        self.update_insight_file_list()
    def update_insight_file_list(self):
        self.file_listbox.delete(0, tk.END)
        insights_path = "Simulation_Data/Insights"
        files = os.listdir(insights_path)
        for file in files:
            self.file_listbox.insert(tk.END, file)

    def on_insight_file_select(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            filename = self.file_listbox.get(selection[0])
            self.show_insight_file_content(filename)

    def show_text_file(self, file_path):
        with open(file_path, "r") as f:
            content = f.read()
        text_widget = tk.Text(self.right_frame, wrap=tk.WORD, font=("Helvetica", 12),
                              bg=self.SECONDARY_BG, fg=self.FG_COLOR)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
    def show_insight_file_content(self, filename):
        file_path = os.path.join("Simulation_Data/Insights", filename)
        if os.path.exists(file_path):
            # Clear previous content
            for widget in self.right_frame.winfo_children():
                widget.destroy()

            if filename.endswith(('.png', '.jpg', '.jpeg')):
                self.show_image(file_path)
            elif filename.endswith('.txt'):
                self.show_text_file(file_path)
            else:
                label = tk.Label(self.right_frame, text="Unsupported file type", fg="red")
                label.pack(padx=10, pady=10)
        else:
            label = tk.Label(self.right_frame, text="File not found", fg="red")
            label.pack(padx=10, pady=10)
    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        files = os.listdir("Simulation_Data")
        for file in files:
            self.file_listbox.insert(tk.END, file)

    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            filename = self.file_listbox.get(selection[0])
            self.show_file_content(filename)

    def show_analysis_content(self, content_type):
        for widget in self.middle_frame.winfo_children():
            widget.destroy()

        if content_type == "View Insights":
            self.show_insights()
        elif content_type == "Fuel Level Distribution":
            self.show_image("Simulation_Data/Insights/fuel_level_distribution.png")
        elif content_type == "Vehicle Position Heatmap":
            self.show_image("Simulation_Data/Insights/vehicle_position_heatmap.png")

    def show_insights(self):
        insights_path = "Simulation_Data/Insights/simulation_insights.txt"
        if os.path.exists(insights_path):
            with open(insights_path, "r") as f:
                insights = f.read()
            text_widget = tk.Text(self.middle_frame, wrap=tk.WORD, font=("Helvetica", 12),
                                  bg=self.SECONDARY_BG, fg=self.FG_COLOR)
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.insert(tk.END, insights)
            text_widget.config(state=tk.DISABLED)
        else:
            label = tk.Label(self.middle_frame, text="Insights file not found", fg="red")
            label.pack(padx=10, pady=10)

    def show_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((800, 600), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(self.right_frame, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)

    def show_file_content(self, filename):
        file_path = os.path.join("Simulation_Data", filename)
        if os.path.exists(file_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                self.show_image(file_path)
            elif filename.endswith('.txt'):
                with open(file_path, "r") as f:
                    content = f.read()
                text_widget = tk.Text(self.middle_frame, wrap=tk.WORD, font=("Helvetica", 12), bg="#2e2e2e", fg="white")
                text_widget.pack(fill=tk.BOTH, expand=True)
                text_widget.insert(tk.END, content)
                text_widget.config(state=tk.DISABLED)
            else:
                label = tk.Label(self.middle_frame, text="Unsupported file type", fg="red")
                label.pack(padx=10, pady=10)
        else:
            label = tk.Label(self.middle_frame, text="File not found", fg="red")
            label.pack(padx=10, pady=10)

    def create_about_tab_widgets(self, about_tab):
        label = tk.Label(about_tab, text="About", font=("Helvetica", 24), fg=self.FG_COLOR, bg=self.BG_COLOR)
        label.pack(pady=50)

        text = tk.Text(about_tab, wrap=tk.WORD, font=("Helvetica", 12), width=80, height=20,
                       bg=self.SECONDARY_BG, fg=self.FG_COLOR, insertbackground=self.FG_COLOR)
        text.pack(padx=40, pady=40)  # Increased padding
        text.insert(tk.END, "Logistics Management App\n")


    def update_video_list(self):
        self.video_listbox.delete(0, tk.END)
        videos = [f for f in os.listdir(self.video_folder) if f.endswith('.mp4')]
        videos.sort(key=lambda x: int(x.split('_')[1]))
        for video in videos:
            self.video_listbox.insert(tk.END, video)

        self.progress_label.config(text=f"{len(videos)} / 15 days generated")

        if len(videos) < 15:
            self.master.after(5000, self.update_video_list)

        if len(videos) > 0 and self.current_video is None:
            self.video_listbox.selection_set(0)
            self.load_video()

    def on_video_select(self, event):
        self.load_video()

    def load_video(self):
        selection = self.video_listbox.curselection()
        if selection:
            video_name = self.video_listbox.get(selection[0])
            self.current_video = cv2.VideoCapture(os.path.join(self.video_folder, video_name))
            self.total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.current_video.get(cv2.CAP_PROP_FPS)
            self.current_frame = 0
            self.progress_bar['maximum'] = self.total_frames
            self.update_time_label()
            self.show_frame()

    def show_frame(self):
        if self.current_video is not None and self.is_playing:
            self.current_video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.current_video.read()
            if ret:
                self.current_frame += 1
                self.progress_bar['value'] = self.current_frame
                self.update_time_label()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame to fit canvas while maintaining aspect ratio
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                frame_height, frame_width = frame.shape[:2]
                aspect_ratio = frame_width / frame_height

                if canvas_width <= 1 or canvas_height <= 1:  # Canvas not properly sized yet
                    self.master.after(100, self.show_frame)  # Try again after a short delay
                    return

                if canvas_width / canvas_height > aspect_ratio:
                    new_width = int(canvas_height * aspect_ratio)
                    new_height = canvas_height
                else:
                    new_width = canvas_width
                    new_height = int(canvas_width / aspect_ratio)

                # Ensure new dimensions are at least 1x1
                new_width = max(1, new_width)
                new_height = max(1, new_height)

                frame = cv2.resize(frame, (new_width, new_height))

                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo, anchor=tk.CENTER)
                self.canvas.image = photo

                self.master.after(int(1000 / self.fps), self.show_frame)
            else:
                self.stop()

    def toggle_play(self):
        if self.current_video is None:
            self.load_video()

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_button.config(text="⏸")
            self.show_frame()
        else:
            self.play_pause_button.config(text="⏵")

    def stop(self):
        self.is_playing = False
        if self.current_video is not None:
            self.current_video.release()
            self.current_video = None
            self.progress_bar['value'] = 0
            self.play_pause_button.config(text="⏵")
            self.update_time_label()

    def backward_10s(self):
        if self.current_video is not None:
            new_frame = max(0, self.current_frame - int(10 * self.fps))
            self.current_frame = new_frame
            self.show_frame()

    def forward_10s(self):
        if self.current_video is not None:
            new_frame = min(self.total_frames - 1, self.current_frame + int(10 * self.fps))
            self.current_frame = new_frame
            self.show_frame()

    def take_screenshot(self):
        if self.current_video is not None:
            ret, frame = self.current_video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
                if file_path:
                    img.save(file_path)
                    print(f"Screenshot saved as {file_path}")

    def on_progress_bar_click(self, event):
        if self.current_video is not None:
            clicked_x = event.x
            total_width = self.progress_bar.winfo_width()
            click_percentage = clicked_x / total_width
            new_frame = int(click_percentage * self.total_frames)
            self.current_frame = new_frame
            self.show_frame()

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.master.attributes("-fullscreen", self.is_fullscreen)
        self.fullscreen_button.config(text="🗗" if self.is_fullscreen else "⛶")

    def update_time_label(self):
        current_time = self.current_frame / self.fps
        total_time = self.total_frames / self.fps
        self.time_label.config(text=f"{self.format_time(current_time)} / {self.format_time(total_time)}")

    def format_time(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"

def start_app():
    root = tk.Tk()
    root.title("Logistics Analysis App")
    root.geometry("1280x720")
    app = VideoPlayer(root)
    root.mainloop()

if __name__ == "__main__":
    start_app()
