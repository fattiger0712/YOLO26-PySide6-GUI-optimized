# YOLO26 PySide6 GUI Optimized

这是从原项目中拆出的新版浅色检测工作台副本，和旧代码目录分开维护。

## 运行

```powershell
pip install -r requirements.txt
python main.py
```

检测历史会写入 `outputs/history.json` 和 `outputs/history.csv`。

左侧“批量处理文件夹”可以递归处理文件夹内的图片和视频，并在历史记录中生成一条批量汇总。
