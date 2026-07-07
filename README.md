# YOLO26 PySide6 GUI Optimized

这是从原项目中拆出的新版浅色检测工作台副本，和旧代码目录分开维护。

## 运行

```powershell
pip install -r requirements.txt
python main.py
```

检测历史会写入 `outputs/history.json` 和 `outputs/history.csv`。

左侧“批量处理文件夹”可以递归处理文件夹内的图片和视频，并在历史记录中生成一条批量汇总。

## 验收可视化

顶部“数据集检查”可以选择 YOLO 数据集目录或 `data.yaml`，输出类别图片数/检测框数/小目标柱状图，以及缺失标注、孤儿标注、损坏图片、类别越界等检查报告。报告保存到 `outputs/dataset_checks/`。

顶部“评测集评估”可以选择自标注评测集，使用当前权重生成预测图、precision、recall、AP/mAP、误检率、漏检率，并将漏检、误检、类别错误、小目标失败图片归档到 `outputs/evaluations/`。

顶部“训练启动”可以从 GUI 配置 data.yaml、基础模型、epochs、imgsz、batch、device 并启动训练；训练完成后会自动导入 `weights/best.pt` 到 `models/`。启动时也会自动扫描外层 `runs/` 里的已有训练结果并导入 best 权重。
