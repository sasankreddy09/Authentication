# ==== 15. Preview (small copy, so the notebook stays light) ====
from IPython.display import Video, display
prev = WORK / "preview.mp4"
ff("-i", PROJ / "render" / "final.mp4", "-vf", "scale=854:-2", "-c:v", "libx264", "-crf", "28",
   "-preset", "veryfast", "-c:a", "aac", "-b:a", "96k", prev)
display(Video(str(prev), embed=True, width=720))
