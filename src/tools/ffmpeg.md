```text
-*- coding: UTF-8 -*-
Author: https://github.com/Guanyan1996
         ┌─┐       ┌─┐
      ┌──┘ ┴───────┘ ┴──┐
      │                 │
      │       ───       │
      │  ─┬┘       └┬─  │
      │                 │
      │       ─┴─       │
      │                 │
      └───┐         ┌───┘
          │         │
          │         │
          │         │
          │         └──────────────┐
          │                        │
          │                        ├─┐
          │                        ┌─┘
          │                        │
          └─┐  ┐  ┌───────┬──┐  ┌──┘
            │ ─┤ ─┤       │ ─┤ ─┤
            └──┴──┘       └──┴──┘
                神兽保佑
                代码无BUG!
```

```shell
### 视频抽所有帧 -r 1/1 会减慢速度
ffmpeg -i "ch01010_20210328144017.mp4.cut.mp4" frames/$filename-%03d.jpg
### 抽取指定帧
ffmpeg -ss 01:23:45 -i input -vframes 1 -q:v 2 output.jpg
### 视频转成RTSP流 esaydarwin
ffmpeg -re -y -i ${input} -c:v copy -rtsp_transport tcp -af arealtime -vcodec h264 -f rtsp ${esaydarwin_server}

注意：ffmpeg操作切勿在dev docker里面完成，docker里ffmpeg有相关问题，转出的视频无法使用

### carsdk的video-capture的硬解部分代码较旧，公司nvr视频需要做处理去掉b帧后跑取数据
ffmpeg -i input.mp4 -bf 0 test_b0.mp4

### 截取视频指定帧(例子为截取前7500帧)
### 不加setpts=PTS-STARTPTS参数的话，截取的视频会保留前面帧号相关存在，内容不存在，视频读取会从截取start帧开始，视频读取出现异常。
### 加了setpts=PTS-STARTPTS参数，视频会清楚之前帧，从截取帧开始
ffmpeg -i input.mp4 -vf "select=between(n\,0\,7499),setpts=PTS-STARTPTS" -y -acodec copy ./output.mp4

### 统计视频I、B、P帧（例子为查看B帧数量)
ffprobe -v quiet -show_frames transcoded123.mp4 | grep "pict_type=B" | wc -l
```
