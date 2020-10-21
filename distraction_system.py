from playsound import playsound  
import vlc
import time
import os

def distract():
    playsound("bipbip.wav")
    playsound("bipbip.wav")
    instance = vlc.Instance()
    media_ply = instance.media_player_new()
    media_ply.set_mrl("bean.mp4")
    media_ply.play()
    time.sleep(5)
    media_ply.stop()