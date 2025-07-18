@echo off
setlocal enabledelayedexpansion

for %%i in (webm-audio\*.webm) do (
    set "filename=%%~ni"
    set "output_file=wav-audio\!filename!.wav"

    if not exist "!output_file!" (
        "C:\Apps\anaconda3\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win64-v4.2.2.exe" -i "%%i" -acodec pcm_s16le -ar 44100 -ac 1 -f wav "!output_file!" && (
            echo Conversion complete: %%~nxi -> !filename!.wav
        ) || (
            echo Error converting: %%~nxi
        )
    ) else (
        echo Skipped: !filename!.wav already exists
    )
)

endlocal