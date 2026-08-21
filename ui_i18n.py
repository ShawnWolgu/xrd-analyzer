"""Centralized Chinese, Japanese, and English UI translations."""

from __future__ import annotations

from typing import Any


SUPPORTED_LANGUAGES = ("zh", "ja", "en")
LANGUAGE_NAMES = {"zh": "中文", "ja": "日本語", "en": "English"}


_T: dict[str, dict[str, str]] = {
    "通用 X 射线衍射分析工具": {
        "ja": "汎用 X 線回折解析ツール",
        "en": "General-purpose X-ray diffraction analysis tool",
    },
    "界面语言:": {"ja": "表示言語:", "en": "Language:"},
    "峰参数配置": {"ja": "ピークパラメータ設定", "en": "Peak Configuration"},
    "可选，例如: Sample(002)": {
        "ja": "任意（例: Sample(002)）",
        "en": "Optional, e.g. Sample(002)",
    },
    "峰名称:": {"ja": "ピーク名:", "en": "Peak name:"},
    "峰中心 (2θ):": {"ja": "ピーク中心 (2θ):", "en": "Peak center (2θ):"},
    "范围下限:": {"ja": "範囲下限:", "en": "Lower bound:"},
    "范围上限:": {"ja": "範囲上限:", "en": "Upper bound:"},
    "峰类型:": {"ja": "ピーク種別:", "en": "Peak type:"},
    "样品峰": {"ja": "試料ピーク", "en": "Sample peak"},
    "基底峰": {"ja": "基板ピーク", "en": "Substrate peak"},
    "就绪": {"ja": "準備完了", "en": "Ready"},
    "1. 数据加载与合并": {"ja": "1. データの読み込みと結合", "en": "1. Data Loading and Merging"},
    "添加数据文件": {"ja": "データファイルを追加", "en": "Add Data Files"},
    "移除选中": {"ja": "選択項目を削除", "en": "Remove Selected"},
    "清空文件": {"ja": "ファイル一覧をクリア", "en": "Clear Files"},
    "重置系统": {"ja": "システムをリセット", "en": "Reset System"},
    "清除所有数据、峰和结果，恢复初始状态": {
        "ja": "すべてのデータ、ピーク、結果を消去して初期状態に戻します",
        "en": "Clear all data, peaks, and results and restore the initial state",
    },
    "加载Excel项目": {"ja": "Excel プロジェクトを読み込む", "en": "Load Excel Project"},
    "读取本程序导出的Excel报告并恢复数据、峰和界面配置": {
        "ja": "本ソフトが出力した Excel レポートからデータ、ピーク、画面設定を復元します",
        "en": "Restore data, peaks, and UI settings from an Excel report exported by this application",
    },
    "2θ范围:": {"ja": "2θ 範囲:", "en": "2θ range:"},
    "应用范围": {"ja": "範囲を適用", "en": "Apply Range"},
    "Y轴显示:": {"ja": "Y 軸表示:", "en": "Y-axis scale:"},
    "Linear": {"ja": "線形", "en": "Linear"},
    "Log": {"ja": "対数", "en": "Log"},
    "Mixed": {"ja": "混合", "en": "Mixed"},
    "2. 数据预处理": {"ja": "2. データ前処理", "en": "2. Data Preprocessing"},
    "滤波器:": {"ja": "フィルター:", "en": "Filter:"},
    "无": {"ja": "なし", "en": "None"},
    "高斯滤波": {"ja": "ガウシアンフィルター", "en": "Gaussian"},
    "FFT滤波": {"ja": "FFT フィルター", "en": "FFT"},
    "窗口长度:": {"ja": "窓長:", "en": "Window length:"},
    "背景扣除:": {"ja": "バックグラウンド除去:", "en": "Background subtraction:"},
    "多项式": {"ja": "多項式", "en": "Polynomial"},
    "多项式阶数:": {"ja": "多項式次数:", "en": "Polynomial degree:"},
    "应用预处理": {"ja": "前処理を適用", "en": "Apply Preprocessing"},
    "重置为原始数据": {"ja": "生データに戻す", "en": "Reset to Raw Data"},
    "4. 拟合配置": {"ja": "4. フィッティング設定", "en": "4. Fitting Configuration"},
    "强制样品峰FWHM相等": {
        "ja": "試料ピークの FWHM を共通化",
        "en": "Constrain Sample-peak FWHM to Be Equal",
    },
    "最小峰间距:": {"ja": "最小ピーク間隔:", "en": "Minimum peak separation:"},
    "拟合方法:": {"ja": "フィッティング手法:", "en": "Fitting method:"},
    "目标函数:": {"ja": "目的関数:", "en": "Objective:"},
    "Log权重:": {"ja": "対数重み:", "en": "Log weight:"},
    "Log强度底值 I₀:": {"ja": "対数強度フロア I₀:", "en": "Log intensity floor I₀:"},
    "Log残差使用log10(I + I₀)；I₀应依据噪声底人工设置": {
        "ja": "対数残差には log10(I + I₀) を使用します。I₀ はノイズフロアに基づいて設定してください",
        "en": "Log residuals use log10(I + I₀); set I₀ manually from the noise floor",
    },
    "包含区间:": {"ja": "含める範囲:", "en": "Include ranges:"},
    "留空=全部；例 42.5-43.5; 54-56": {
        "ja": "空欄=全範囲（例: 42.5-43.5; 54-56）",
        "en": "Blank = all; e.g. 42.5-43.5; 54-56",
    },
    "排除区间:": {"ja": "除外範囲:", "en": "Exclude ranges:"},
    "例 42.8-43.2": {"ja": "例: 42.8-43.2", "en": "e.g. 42.8-43.2"},
    "背景强度:": {"ja": "バックグラウンド強度:", "en": "Background intensity:"},
    "设置固定的背景强度值或显示拟合后的背景值": {
        "ja": "固定バックグラウンド強度を設定、またはフィッティング値を表示します",
        "en": "Set a fixed background intensity or display the fitted value",
    },
    "固定": {"ja": "固定", "en": "Fixed"},
    "勾选此项将强制使用设定的背景值，不进行拟合": {
        "ja": "オンにすると設定値を固定し、バックグラウンドを最適化しません",
        "en": "Use the specified background value without optimizing it",
    },
    "执行拟合": {"ja": "フィッティングを実行", "en": "Run Fit"},
    "基于当前结果优化拟合 (Refine)": {
        "ja": "現在の結果から再フィッティング (Refine)",
        "en": "Refine from Current Result",
    },
    "使用上一次的拟合结果作为初始值，并应用表格中的锁定设置进行再拟合": {
        "ja": "前回の結果を初期値として、表の固定設定を適用して再フィッティングします",
        "en": "Refit using the previous result as the initial values and the locks set in the table",
    },
    "接受当前结果作为下一轮初值": {
        "ja": "現在の結果を次回の初期値として採用",
        "en": "Accept Current Result as Next Initial Guess",
    },
    "人工确认当前候选结果，并保存center、area、FWHM和η作为下一轮初值": {
        "ja": "候補結果を確認し、center、area、FWHM、η を次回の初期値として保存します",
        "en": "Accept the candidate and save center, area, FWHM, and η as the next initial values",
    },
    "5. 结果导出": {"ja": "5. 結果の出力", "en": "5. Result Export"},
    "导出Excel报告": {"ja": "Excel レポートを出力", "en": "Export Excel Report"},
    "导出高清图片": {"ja": "高解像度画像を出力", "en": "Export High-resolution Figure"},
    "3. 峰识别与添加": {"ja": "3. ピークの指定と追加", "en": "3. Peak Definition and Addition"},
    "2θ位置 (°)": {"ja": "2θ 位置 (°)", "en": "2θ position (°)"},
    "理论晶面间距 d (Å)": {"ja": "理論面間隔 d (Å)", "en": "Theoretical d-spacing (Å)"},
    "峰名，例如 Sample(002)": {
        "ja": "ピーク名（例: Sample(002)）",
        "en": "Peak name, e.g. Sample(002)",
    },
    "添加": {"ja": "追加", "en": "Add"},
    "X射线波长 λ:": {"ja": "X 線波長 λ:", "en": "X-ray wavelength λ:"},
    "项目统一使用的X射线波长；默认 Cu Kα1 = 1.5406 Å": {
        "ja": "プロジェクト共通の X 線波長。既定値: Cu Kα1 = 1.5406 Å",
        "en": "Project-wide X-ray wavelength; default Cu Kα1 = 1.5406 Å",
    },
    "默认 Cu Kα1": {"ja": "既定値 Cu Kα1", "en": "Default Cu Kα1"},
    "导入峰位置 (.txt)": {"ja": "ピーク位置を読み込む (.txt)", "en": "Import Peak Positions (.txt)"},
    "导出拟合峰位置 (.txt)": {"ja": "フィッティング峰位置を出力 (.txt)", "en": "Export Fitted Peak Positions (.txt)"},
    "手动添加峰 (点击图上)": {"ja": "手動でピークを追加（図をクリック）", "en": "Add Peak Manually (Click Plot)"},
    "手动添加峰 (已激活)": {"ja": "手動ピーク追加（有効）", "en": "Manual Peak Addition (Active)"},
    "峰值拟合管理": {"ja": "ピークフィッティング管理", "en": "Peak Fitting Management"},
    "名称": {"ja": "名称", "en": "Name"},
    "位置 (Pos)": {"ja": "位置 (Pos)", "en": "Position (Pos)"},
    "面积": {"ja": "面積", "en": "Area"},
    "高度 (Height)": {"ja": "高さ (Height)", "en": "Height"},
    "峰状态": {"ja": "ピーク状態", "en": "Peak state"},
    "类型": {"ja": "種別", "en": "Type"},
    "拟合Pseudo-Voigt分布的积分面积（lmfit amplitude）": {
        "ja": "フィッティングした Pseudo-Voigt 分布の積分面積（lmfit amplitude）",
        "en": "Integrated area of the fitted Pseudo-Voigt distribution (lmfit amplitude)",
    },
    "Pseudo-Voigt半高全宽（2θ度）。样品峰允许0.02–3.00°，基底峰允许0.02–2.00°；勾选后按输入值精确固定。": {
        "ja": "Pseudo-Voigt の半値全幅（2θ 度）。試料ピーク: 0.02–3.00°、基板ピーク: 0.02–2.00°。チェックすると入力値に固定します。",
        "en": "Pseudo-Voigt FWHM in degrees 2θ. Sample peaks: 0.02–3.00°; substrate peaks: 0.02–2.00°. Check to fix the entered value.",
    },
    "Pseudo-Voigt Lorentzian比例，范围0–1": {
        "ja": "Pseudo-Voigt の Lorentzian 比率（0–1）",
        "en": "Pseudo-Voigt Lorentzian fraction, range 0–1",
    },
    "优化=参数参与拟合；冻结=固定完整峰形；禁用=本轮峰分量为零": {
        "ja": "最適化=パラメータを最適化、凍結=ピーク形状を固定、無効=今回の成分をゼロ",
        "en": "Optimize = vary parameters; Frozen = fix the complete peak shape; Disabled = zero contribution",
    },
    "整体平移 2θ:": {"ja": "2θ 一括シフト:", "en": "Global 2θ shift:"},
    "所有峰向左移动一个步长": {"ja": "全ピークを 1 ステップ左へ移動", "en": "Shift all peaks left by one step"},
    "拖动后释放：按滑块格数×步长整体平移所有峰，随后自动回中": {
        "ja": "ドラッグして離すと、目盛数×ステップで全ピークを移動し、中央に戻ります",
        "en": "Drag and release to shift all peaks by slider steps × step size; the slider then returns to center",
    },
    "所有峰向右移动一个步长": {"ja": "全ピークを 1 ステップ右へ移動", "en": "Shift all peaks right by one step"},
    "步长:": {"ja": "ステップ:", "en": "Step:"},
    "峰位整体平移步长，单位为degree 2θ": {
        "ja": "ピーク位置の一括シフト量（degree 2θ）",
        "en": "Global peak-position shift step in degrees 2θ",
    },
    "撤销": {"ja": "元に戻す", "en": "Undo"},
    "回到上一次已完成的拟合结果（最多5步）": {
        "ja": "前の完了済みフィッティング結果へ戻る（最大 5 件）",
        "en": "Return to the previous completed fit result (up to 5 results)",
    },
    "恢复": {"ja": "やり直す", "en": "Redo"},
    "恢复下一次已完成的拟合结果": {"ja": "次の完了済み結果へ進む", "en": "Restore the next completed fit result"},
    "删除选中峰": {"ja": "選択ピークを削除", "en": "Delete Selected Peaks"},
    "清除所有峰": {"ja": "全ピークを消去", "en": "Clear All Peaks"},
    "数据与拟合": {"ja": "データとフィッティング", "en": "Data and Fitting"},
    "优化": {"ja": "最適化", "en": "Optimize"},
    "冻结": {"ja": "凍結", "en": "Frozen"},
    "禁用": {"ja": "無効", "en": "Disabled"},
    "0=Gaussian，1=Lorentzian；冻结时作为完整峰形的一部分固定": {
        "ja": "0=Gaussian、1=Lorentzian。凍結時はピーク形状の一部として固定されます",
        "en": "0 = Gaussian, 1 = Lorentzian; fixed as part of the complete frozen peak shape",
    },
    "允许范围：{lower:.2f}–{upper:.2f}°（2θ）。未勾选时作为拟合初值；勾选时精确固定为该值。": {
        "ja": "許容範囲: {lower:.2f}–{upper:.2f}° (2θ)。未チェック時は初期値、チェック時は入力値に固定します。",
        "en": "Allowed range: {lower:.2f}–{upper:.2f}° (2θ). Unchecked: initial guess; checked: fixed at the entered value.",
    },
    "一级布拉格条件下的理论晶面间距d，不是晶格常数": {
        "ja": "一次 Bragg 条件の理論面間隔 d（格子定数ではありません）",
        "en": "Theoretical first-order Bragg d-spacing, not a lattice constant",
    },
    "衍射峰中心位置2θ，单位为度": {
        "ja": "回折ピーク中心 2θ（度）",
        "en": "Diffraction peak center 2θ in degrees",
    },
    "自定义波长": {"ja": "ユーザー指定波長", "en": "Custom wavelength"},
    "可用": {"ja": "あり", "en": "Available"},
    "不可用": {"ja": "なし", "en": "Unavailable"},
    "右": {"ja": "右", "en": "right"},
    "左": {"ja": "左", "en": "left"},
    # Dialog titles and messages.
    "警告": {"ja": "警告", "en": "Warning"},
    "错误": {"ja": "エラー", "en": "Error"},
    "成功": {"ja": "完了", "en": "Success"},
    "加载Excel项目将替换当前数据和峰设置，是否继续？": {
        "ja": "Excel プロジェクトを読み込むと現在のデータとピーク設定が置き換わります。続行しますか？",
        "en": "Loading the Excel project will replace the current data and peak settings. Continue?",
    },
    "替换当前项目": {"ja": "現在のプロジェクトを置換", "en": "Replace Current Project"},
    "项目加载失败": {"ja": "プロジェクトの読み込みに失敗", "en": "Project Load Failed"},
    "拟合进行中": {"ja": "フィッティング実行中", "en": "Fitting in Progress"},
    "请等待当前拟合结束后再重置系统": {
        "ja": "現在のフィッティング完了後にシステムをリセットしてください",
        "en": "Wait for the current fit to finish before resetting the system",
    },
    "确认重置": {"ja": "リセットの確認", "en": "Confirm Reset"},
    "确定要重置系统吗？\n这将清除所有已加载的数据、峰设置和拟合结果。": {
        "ja": "システムをリセットしますか？\n読み込んだデータ、ピーク設定、フィッティング結果はすべて消去されます。",
        "en": "Reset the system?\nThis will clear all loaded data, peak settings, and fit results.",
    },
    "未添加任何新文件（可能格式错误或已存在）": {
        "ja": "新しいファイルは追加されませんでした（形式エラーまたは追加済みの可能性があります）",
        "en": "No new files were added (they may be invalid or already loaded)",
    },
    "范围错误": {"ja": "範囲エラー", "en": "Range Error"},
    "2θ范围下限必须小于上限": {"ja": "2θ 範囲の下限は上限より小さくしてください", "en": "The lower 2θ bound must be smaller than the upper bound"},
    "所选2θ范围内没有数据": {"ja": "選択した 2θ 範囲にデータがありません", "en": "No data lie within the selected 2θ range"},
    "不能冻结峰形": {"ja": "ピーク形状を凍結できません", "en": "Cannot Freeze Peak Shape"},
    "请先拟合并点击“接受当前结果作为下一轮初值”。": {
        "ja": "先にフィッティングを行い、「現在の結果を次回の初期値として採用」をクリックしてください。",
        "en": "Fit first, then click “Accept Current Result as Next Initial Guess.”",
    },
    "没有数据": {"ja": "データがありません", "en": "No Data"},
    "请先加载XRD数据": {"ja": "先に XRD データを読み込んでください", "en": "Load XRD data first"},
    "输入无效": {"ja": "入力が無効です", "en": "Invalid Input"},
    "峰位超出数据范围": {"ja": "ピーク位置がデータ範囲外です", "en": "Peak Outside Data Range"},
    "导入失败": {"ja": "読み込みに失敗", "en": "Import Failed"},
    "没有拟合结果": {"ja": "フィッティング結果がありません", "en": "No Fit Result"},
    "请先完成一次拟合": {"ja": "先にフィッティングを完了してください", "en": "Complete a fit first"},
    "没有可导出的峰": {"ja": "出力可能なピークがありません", "en": "No Peaks to Export"},
    "当前拟合结果中没有有效峰位": {"ja": "現在の結果に有効なピーク位置がありません", "en": "The current fit result contains no valid peak positions"},
    "导出失败": {"ja": "出力に失敗", "en": "Export Failed"},
    "没有峰": {"ja": "ピークがありません", "en": "No Peaks"},
    "请先导入或添加峰": {"ja": "先にピークを読み込むか追加してください", "en": "Import or add peaks first"},
    "无法平移峰位": {"ja": "ピーク位置をシフトできません", "en": "Cannot Shift Peaks"},
    "请等待当前拟合结束后再删除峰": {"ja": "現在のフィッティング完了後にピークを削除してください", "en": "Wait for the current fit to finish before deleting peaks"},
    "请选择要删除的峰": {"ja": "削除するピークを選択してください", "en": "Select peaks to delete"},
    "请先添加峰": {"ja": "先にピークを追加してください", "en": "Add peaks first"},
    "没有候选结果": {"ja": "候補結果がありません", "en": "No Candidate Result"},
    "结果未收敛": {"ja": "結果が収束していません", "en": "Result Did Not Converge"},
    "求解器没有报告成功。仍然接受这个候选结果作为下一轮初值吗？": {
        "ja": "ソルバーは成功を報告していません。この候補を次回の初期値として採用しますか？",
        "en": "The solver did not report success. Accept this candidate as the next initial guess anyway?",
    },
    "拟合区间格式错误": {"ja": "フィッティング範囲の形式エラー", "en": "Invalid Fit-range Format"},
    "候选拟合结果": {"ja": "フィッティング候補結果", "en": "Candidate Fit Result"},
    "候选结果需要检查": {"ja": "候補結果を確認してください", "en": "Candidate Result Requires Review"},
    "拟合错误": {"ja": "フィッティングエラー", "en": "Fitting Error"},
    "请先完成拟合": {"ja": "先にフィッティングを完了してください", "en": "Complete a fit first"},
    "保存Excel报告": {"ja": "Excel レポートを保存", "en": "Save Excel Report"},
    "保存图片": {"ja": "画像を保存", "en": "Save Figure"},
    "选择XRD数据文件": {"ja": "XRD データファイルを選択", "en": "Select XRD Data Files"},
    "项目: {filename}": {"ja": "プロジェクト: {filename}", "en": "Project: {filename}"},
    "导入峰列表": {"ja": "ピークリストを読み込む", "en": "Import Peak List"},
    "导出拟合峰位置": {"ja": "フィッティング峰位置を出力", "en": "Export Fitted Peak Positions"},
    "Excel项目 (*.xlsx *.xls)": {"ja": "Excel プロジェクト (*.xlsx *.xls)", "en": "Excel Projects (*.xlsx *.xls)"},
    "文本文件 (*.txt *.TXT)": {"ja": "テキストファイル (*.txt *.TXT)", "en": "Text Files (*.txt *.TXT)"},
    "文本文件 (*.txt);;所有文件 (*)": {"ja": "テキストファイル (*.txt);;すべてのファイル (*)", "en": "Text Files (*.txt);;All Files (*)"},
    "Excel文件 (*.xlsx)": {"ja": "Excel ファイル (*.xlsx)", "en": "Excel Files (*.xlsx)"},
    "PNG文件 (*.png);;PDF文件 (*.pdf);;SVG文件 (*.svg)": {
        "ja": "PNG ファイル (*.png);;PDF ファイル (*.pdf);;SVG ファイル (*.svg)",
        "en": "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
    },
    "FWHM超出范围": {"ja": "FWHM が範囲外です", "en": "FWHM Outside Allowed Range"},
    "η超出范围": {"ja": "η が範囲外です", "en": "η Outside Allowed Range"},
    "Pseudo-Voigt Lorentzian比例η必须在0和1之间。": {
        "ja": "Pseudo-Voigt の Lorentzian 比率 η は 0～1 にしてください。",
        "en": "The Pseudo-Voigt Lorentzian fraction η must be between 0 and 1.",
    },
    "X射线波长必须是大于0的有限值": {
        "ja": "X 線波長は正の有限値にしてください",
        "en": "The X-ray wavelength must be a positive finite value",
    },
    "2θ必须位于0°到180°之间": {
        "ja": "2θ は 0°～180° の範囲にしてください",
        "en": "2θ must lie between 0° and 180°",
    },
    "理论晶面间距d必须是大于0的有限值": {
        "ja": "理論面間隔 d は正の有限値にしてください",
        "en": "The theoretical d-spacing must be a positive finite value",
    },
    "理论晶面间距d小于λ/2，无法满足一级布拉格条件": {
        "ja": "理論面間隔 d が λ/2 より小さいため、一次 Bragg 条件を満たせません",
        "en": "The theoretical d-spacing is smaller than λ/2 and cannot satisfy the first-order Bragg condition",
    },
    "峰位平移量必须是有限的2θ角度": {
        "ja": "ピークシフト量は有限の 2θ 角度にしてください",
        "en": "The peak shift must be a finite 2θ angle",
    },
    "当前XRD数据不包含有效的2θ坐标": {
        "ja": "現在の XRD データに有効な 2θ 座標がありません",
        "en": "The current XRD data contain no valid 2θ coordinates",
    },
    "Log强度底值必须为有限正数": {
        "ja": "対数強度フロア I₀ は正の有限値にしてください",
        "en": "The log intensity floor I₀ must be a positive finite value",
    },
    "当前拟合数据包含负强度，不能使用Log目标": {
        "ja": "現在のフィッティングデータに負の強度が含まれるため、Log 目的関数は使用できません",
        "en": "The current fit data contain negative intensities, so the Log objective cannot be used",
    },
    "当前拟合数据包含负强度，不能使用Log或Mixed目标": {
        "ja": "現在のフィッティングデータに負の強度が含まれるため、Log または Mixed 目的関数は使用できません",
        "en": "The current fit data contain negative intensities, so Log and Mixed objectives cannot be used",
    },
    "当前模型产生负强度，不能使用Log目标": {
        "ja": "現在のモデルが負の強度を生成するため、Log 目的関数は使用できません",
        "en": "The current model produces negative intensities, so the Log objective cannot be used",
    },
    "人工拟合区间没有包含任何数据点": {
        "ja": "指定したフィッティング範囲にデータ点がありません",
        "en": "The specified fit ranges contain no data points",
    },
    # Dynamic status and message templates.
    "Excel项目已恢复: {filename}": {"ja": "Excel プロジェクトを復元しました: {filename}", "en": "Excel project restored: {filename}"},
    "成功添加 {count} 个文件": {"ja": "{count} 個のファイルを追加しました", "en": "Added {count} file(s)"},
    "文件已移除": {"ja": "ファイルを削除しました", "en": "File removed"},
    "列表已清空": {"ja": "一覧をクリアしました", "en": "List cleared"},
    "系统已重置，就绪": {"ja": "システムをリセットしました。準備完了", "en": "System reset; ready"},
    "数据合并失败:\n{error}": {"ja": "データ結合に失敗しました:\n{error}", "en": "Data merge failed:\n{error}"},
    "项目数据已裁剪至 {lower:.3f}–{upper:.3f}° (2θ){removed}": {
        "ja": "プロジェクトデータを {lower:.3f}–{upper:.3f}° (2θ) に切り詰めました{removed}",
        "en": "Project data cropped to {lower:.3f}–{upper:.3f}° (2θ){removed}",
    },
    "，移除 {count} 个范围外峰": {"ja": "、範囲外の {count} ピークを削除", "en": "; removed {count} out-of-range peak(s)"},
    "预处理完成": {"ja": "前処理が完了しました", "en": "Preprocessing complete"},
    "已重置为原始数据": {"ja": "生データに戻しました", "en": "Reset to raw data"},
    "点击图上添加峰...": {"ja": "図をクリックしてピークを追加...", "en": "Click the plot to add a peak..."},
    "已{action}到对应拟合结果": {"ja": "対応する結果へ{action}しました", "en": "{action} to the corresponding fit result"},
    "峰参数已修改，请重新拟合": {"ja": "ピークパラメータを変更しました。再フィッティングしてください", "en": "Peak parameters changed; refit required"},
    "计算得到的峰位 2θ = {center:.6f}°，不在当前数据范围 {lower:.6f}–{upper:.6f}° 内。": {
        "ja": "計算したピーク位置 2θ = {center:.6f}° はデータ範囲 {lower:.6f}–{upper:.6f}° 外です。",
        "en": "Calculated peak position 2θ = {center:.6f}° is outside the data range {lower:.6f}–{upper:.6f}°.",
    },
    "已添加峰: d = {d:.6f} Å, λ = {wavelength:.6f} Å → 2θ = {center:.6f}° {name}": {
        "ja": "ピークを追加: d = {d:.6f} Å, λ = {wavelength:.6f} Å → 2θ = {center:.6f}° {name}",
        "en": "Peak added: d = {d:.6f} Å, λ = {wavelength:.6f} Å → 2θ = {center:.6f}° {name}",
    },
    "已添加峰: 2θ = {center:.6f}° {name}": {"ja": "ピークを追加: 2θ = {center:.6f}° {name}", "en": "Peak added: 2θ = {center:.6f}° {name}"},
    "成功导入 {count} 个峰{skipped}": {"ja": "{count} 個のピークを読み込みました{skipped}", "en": "Imported {count} peak(s){skipped}"},
    " (已忽略 {count} 个超出范围的峰)": {"ja": "（範囲外の {count} ピークを無視）", "en": " ({count} out-of-range peak(s) ignored)"},
    "错误: {error}": {"ja": "エラー: {error}", "en": "Error: {error}"},
    "已导出 {count} 个拟合峰位: {filename}": {"ja": "{count} 個のフィッティング峰位置を出力しました: {filename}", "en": "Exported {count} fitted peak position(s): {filename}"},
    "{count} 个峰已向{direction}平移 {delta:.6f}° (2θ)": {"ja": "{count} 個のピークを{direction}へ {delta:.6f}° (2θ) シフトしました", "en": "Shifted {count} peak(s) {direction} by {delta:.6f}° (2θ)"},
    "峰已删除，请重新拟合": {"ja": "ピークを削除しました。再フィッティングしてください", "en": "Peak deleted; refit required"},
    "正在优化拟合...": {"ja": "再フィッティング中...", "en": "Refining fit..."},
    "已接受当前结果，可冻结峰形或继续优化": {"ja": "現在の結果を採用しました。ピーク形状を凍結するか再最適化できます", "en": "Current result accepted; freeze peak shapes or continue refining"},
    "正在拟合...": {"ja": "フィッティング中...", "en": "Fitting..."},
    "拟合完成，结果尚未接受": {"ja": "フィッティング完了。結果は未採用です", "en": "Fit complete; result not yet accepted"},
    "拟合结束，但结果需要检查": {"ja": "フィッティング終了。結果を確認してください", "en": "Fit finished; result requires review"},
    "拟合失败": {"ja": "フィッティングに失敗しました", "en": "Fit failed"},
    "拟合过程中出错:\n{error}": {"ja": "フィッティング中にエラーが発生しました:\n{error}", "en": "An error occurred during fitting:\n{error}"},
    "求解器消息：{message}": {"ja": "ソルバーのメッセージ: {message}", "en": "Solver message: {message}"},
    "函数评估次数：{count}": {"ja": "関数評価回数: {count}", "en": "Function evaluations: {count}"},
    "协方差：{availability}": {"ja": "共分散: {availability}", "en": "Covariance: {availability}"},
    "边界命中：{items}": {"ja": "境界に到達: {items}", "en": "Bounds reached: {items}"},
    "数值警告：{items}": {"ja": "数値警告: {items}", "en": "Numerical warnings: {items}"},
    "\n\n请检查后点击“接受当前结果”。": {
        "ja": "\n\n確認後、「現在の結果を次回の初期値として採用」をクリックしてください。",
        "en": "\n\nReview the result, then click “Accept Current Result as Next Initial Guess.”",
    },
    "\n\n该结果不会自动成为下一轮初值。": {
        "ja": "\n\nこの結果は次回の初期値として自動採用されません。",
        "en": "\n\nThis result will not automatically become the next initial guess.",
    },
    "{peak_type}峰的FWHM必须在 {lower:.2f}–{upper:.2f}°（2θ）之间。": {
        "ja": "{peak_type}ピークの FWHM は {lower:.2f}–{upper:.2f}° (2θ) の範囲にしてください。",
        "en": "The FWHM of a {peak_type} peak must be between {lower:.2f} and {upper:.2f}° (2θ).",
    },
    "结果已导出至:\n{path}": {"ja": "結果を出力しました:\n{path}", "en": "Results exported to:\n{path}"},
    "导出失败:\n{error}": {"ja": "出力に失敗しました:\n{error}", "en": "Export failed:\n{error}"},
    "图片已保存至:\n{path}": {"ja": "画像を保存しました:\n{path}", "en": "Figure saved to:\n{path}"},
    "保存失败:\n{error}": {"ja": "保存に失敗しました:\n{error}", "en": "Save failed:\n{error}"},
}


def translate(source: str, language: str, **values: Any) -> str:
    """Translate one canonical Chinese UI string and format named values."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported UI language: {language}")
    template = source if language == "zh" else _T.get(source, {}).get(language, source)
    return template.format(**values) if values else template


def has_translation(source: str) -> bool:
    """Return whether a canonical UI string is registered for translation."""
    return source in _T
