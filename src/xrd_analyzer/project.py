"""可恢复 XRD Excel 项目工作簿的读取适配器。"""

from __future__ import annotations

import ast
import json
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pandas as pd


class ProjectWorkbook:
    """读取项目数据、峰、拟合配置和 GUI 状态。"""

    @staticmethod
    def _clean_value(value):
        if isinstance(value, np.generic):
            value = value.item()
        if pd.isna(value):
            return None
        return value

    @classmethod
    def _record(cls, dataframe: pd.DataFrame) -> Dict:
        if dataframe.empty:
            return {}
        return {
            key: cls._clean_value(value)
            for key, value in dataframe.iloc[0].to_dict().items()
        }

    @staticmethod
    def _decode_configuration_value(value):
        if not isinstance(value, str):
            return value
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    @classmethod
    def load(cls, input_path: str) -> Dict:
        """读取工作簿中的数据、峰、拟合配置和项目状态。"""
        workbook = pd.ExcelFile(input_path)
        if "Full_Data" not in workbook.sheet_names:
            raise ValueError("Excel报告缺少Full_Data工作表，不能恢复项目")

        full_data = pd.read_excel(workbook, sheet_name="Full_Data")
        if "2theta" not in full_data.columns:
            raise ValueError("Full_Data缺少2theta列")
        processed_column = next(
            (
                name
                for name in ("Processed_Intensity", "Intensity", "Original_Intensity")
                if name in full_data.columns
            ),
            None,
        )
        if processed_column is None:
            raise ValueError("Full_Data缺少可恢复的强度列")

        x_data = full_data["2theta"].to_numpy(dtype=float)
        processed = full_data[processed_column].to_numpy(dtype=float)
        raw = (
            full_data["Original_Intensity"].to_numpy(dtype=float)
            if "Original_Intensity" in full_data.columns
            else processed.copy()
        )
        fitted = None
        if "Fitted_Intensity" in full_data.columns:
            fitted_values = full_data["Fitted_Intensity"].to_numpy(dtype=float)
            if not np.all(np.isnan(fitted_values)):
                fitted = fitted_values
        background = None
        if "Background" in full_data.columns:
            background_values = full_data["Background"].to_numpy(dtype=float)
            if not np.all(np.isnan(background_values)):
                background = background_values

        peaks = []
        if "Peak_Parameters" in workbook.sheet_names:
            peak_table = pd.read_excel(
                workbook,
                sheet_name="Peak_Parameters",
                converters={
                    "Name": lambda value: str(value) if value is not None else "",
                    "Type": lambda value: str(value) if value is not None else "",
                },
            )
            for record in peak_table.to_dict(orient="records"):
                peaks.append(
                    {
                        key: cls._clean_value(value)
                        for key, value in record.items()
                    }
                )

        project_state = {}
        if "Project_State" in workbook.sheet_names:
            state_table = pd.read_excel(workbook, sheet_name="Project_State")
            if {"Key", "Value_JSON"}.issubset(state_table.columns):
                for _, row in state_table.iterrows():
                    stored_value = row["Value_JSON"]
                    project_state[str(row["Key"])] = (
                        json.loads(stored_value)
                        if isinstance(stored_value, str)
                        else cls._clean_value(stored_value)
                    )

        fit_config = {}
        if "Fit_Configuration" in workbook.sheet_names:
            fit_config = cls._record(
                pd.read_excel(workbook, sheet_name="Fit_Configuration")
            )
            fit_config = {
                key: cls._decode_configuration_value(value)
                for key, value in fit_config.items()
            }

        metrics = {}
        if "Fit_Metrics" in workbook.sheet_names:
            metrics = cls._record(pd.read_excel(workbook, sheet_name="Fit_Metrics"))

        source_files = []
        source_datasets = []
        if "Source_Files" in workbook.sheet_names:
            source_table = pd.read_excel(workbook, sheet_name="Source_Files")
            if "Path" in source_table.columns:
                source_files = [str(path) for path in source_table["Path"].dropna()]
            if {"Path", "Sheet_Name"}.issubset(source_table.columns):
                for _, row in source_table.iterrows():
                    sheet_name = cls._clean_value(row["Sheet_Name"])
                    path = cls._clean_value(row["Path"])
                    if not sheet_name or sheet_name not in workbook.sheet_names:
                        continue
                    source_data = pd.read_excel(workbook, sheet_name=sheet_name)
                    if {"2theta", "Intensity"}.issubset(source_data.columns):
                        source_datasets.append(
                            (
                                str(path),
                                source_data["2theta"].to_numpy(dtype=float),
                                source_data["Intensity"].to_numpy(dtype=float),
                            )
                        )

        if not project_state:
            project_state = {
                "schema_version": 0,
                "range_min": float(np.min(x_data)),
                "range_max": float(np.max(x_data)),
            }
        if peaks and "wavelength_angstrom" not in project_state:
            stored_wavelength = peaks[0].get("Wavelength_Angstrom")
            if stored_wavelength is not None:
                project_state["wavelength_angstrom"] = float(stored_wavelength)
            stored_label = peaks[0].get("Radiation_Label")
            if stored_label:
                project_state["radiation_label"] = str(stored_label)

        return {
            "schema_version": project_state.get("schema_version", 0),
            "x_data": x_data,
            "processed_intensity": processed,
            "raw_intensity": raw,
            "fitted_intensity": fitted,
            "background": background,
            "peaks": peaks,
            "project_state": project_state,
            "fit_config": fit_config,
            "metrics": metrics,
            "source_files": source_files,
            "source_datasets": source_datasets,
        }


class RestoredFitResult(SimpleNamespace):
    """从项目工作簿恢复的最小 lmfit 结果兼容对象。"""

    restored = True
