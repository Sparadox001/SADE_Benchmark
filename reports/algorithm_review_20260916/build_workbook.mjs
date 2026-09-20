import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const reportDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(reportDir, "../..");
const outputDir = path.join(projectDir, "outputs/algorithm_review_20260916");
const data = JSON.parse(await fs.readFile(path.join(reportDir, "workbook_data.json"), "utf8"));

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const versions = workbook.worksheets.add("Version history");
const runs = workbook.worksheets.add("Run data");
const historical = workbook.worksheets.add("Historical 25-run");

const font = "Arial";
const dark = "#1F4E78";
const mid = "#D9EAF7";
const light = "#F3F6F8";
const green = "#E2F0D9";
const blue = "#DDEBF7";
const amber = "#FFF2CC";
const red = "#FCE4D6";
const border = "#B7C9D6";

function styleTitle(sheet, cell, text, endColumn) {
  sheet.getRange(cell).values = [[text]];
  const row = Number(cell.match(/\d+/)[0]);
  const startColumn = cell.match(/[A-Z]+/)[0];
  const range = sheet.getRange(`${startColumn}${row}:${endColumn}${row}`);
  range.format.font = { name: font, size: 15, bold: true, color: "#1F1F1F" };
  range.format.borders = { bottom: { style: "thin", color: dark } };
  range.format.rowHeight = 26;
}

function styleHeader(range) {
  range.format.fill = dark;
  range.format.font = { name: font, size: 10, bold: true, color: "#FFFFFF" };
  range.format.horizontalAlignment = "center";
  range.format.verticalAlignment = "center";
  range.format.wrapText = true;
  range.format.borders = { preset: "inside", style: "thin", color: "#FFFFFF" };
  range.format.rowHeight = 30;
}

function styleBody(range) {
  range.format.font = { name: font, size: 10, color: "#222222" };
  range.format.verticalAlignment = "center";
  range.format.borders = {
    insideHorizontal: { style: "thin", color: "#E6E6E6" },
    bottom: { style: "thin", color: border },
  };
}

for (const sheet of [summary, versions, runs, historical]) {
  sheet.showGridLines = false;
}

// Summary
styleTitle(summary, "A2", "SADE版本实验整理", "L");
for (let i = 0; i < data.notes.length; i += 1) {
  const row = 4 + i;
  summary.mergeCells(`A${row}:L${row}`);
  summary.getRange(`A${row}`).values = [[data.notes[i]]];
}
summary.getRange("A4:L8").format.font = { name: font, size: 10, color: "#555555", italic: true };
summary.getRange("A4:L8").format.wrapText = false;
summary.getRange("A4:L8").format.rowHeight = 20;

summary.getRange("A10").values = [["当前5-run总体配对概览"]];
summary.getRange("A10:L10").format.font = { name: font, size: 12, bold: true, color: dark };
const algorithmHeaders = [
  "算法", "实例数", "Seeds", "运行-实例单元", "成功单元", "成功率", "平均运行rank",
  "胜DSI", "负DSI", "平DSI", "胜原SADE", "负原SADE",
];
summary.getRange("A11:L11").values = [algorithmHeaders];
styleHeader(summary.getRange("A11:L11"));
const algorithmRows = data.key_5seed_algorithm_summary.map((row) => [
  row.algorithm, Number(row.instances), Number(row.seeds), Number(row.run_case_units),
  Number(row.successful_units), Number(row.success_rate), Number(row.average_run_rank),
  Number(row.wins_vs_dsi), Number(row.losses_vs_dsi), Number(row.ties_vs_dsi),
  Number(row.wins_vs_sade), Number(row.losses_vs_sade),
]);
summary.getRange(`A12:L${11 + algorithmRows.length}`).values = algorithmRows;
styleBody(summary.getRange(`A12:L${11 + algorithmRows.length}`));
summary.getRange(`F12:F${11 + algorithmRows.length}`).format.numberFormat = "0.0%";
summary.getRange(`G12:G${11 + algorithmRows.length}`).format.numberFormat = "0.000";
summary.getRange("A12:L12").format.fill = blue;
summary.getRange("A13:L13").format.fill = green;
summary.getRange("A14:L14").format.fill = amber;

const lookup = new Map(data.key_5seed_instance_summary.map((row) => [
  `${row.algorithm}|${row.problem}|${row.dimension}`, row,
]));
const algorithms = ["sade", "dsi", "sade_v2_7", "sade_v2_8", "sade_v2_9"];
const cases = [["c01", 10], ["c01", 30], ["c05", 10], ["c05", 30], ["c20", 10], ["c20", 30], ["c22", 10], ["c22", 30]];
function display(row) {
  const sr = Number(row.success_rate);
  if (sr < 1) return `${(100 * sr).toFixed(0)}%`;
  const scientific = (value) => Number(value).toExponential(2).replace(/e([+-])(\d)$/, "e$10$2");
  return `${scientific(row.feasible_objective_mean)} (${scientific(row.feasible_objective_std)})`;
}
summary.getRange("A19").values = [["当前代码5-run实例结果"]];
summary.getRange("A19:F19").format.font = { name: font, size: 12, bold: true, color: dark };
summary.getRange("A20:F20").values = [["实例", "原SADE", "DSI", "V2.7", "V2.8", "V2.9"]];
styleHeader(summary.getRange("A20:F20"));
const instanceRows = cases.map(([problem, dimension]) => [
  `${String(problem).toUpperCase()}-${dimension}D`,
  ...algorithms.map((algorithm) => display(lookup.get(`${algorithm}|${problem}|${dimension}`))),
]);
summary.getRange("A21:F28").values = instanceRows;
styleBody(summary.getRange("A21:F28"));
summary.getRange("B21:F28").format.horizontalAlignment = "right";

summary.getRange("H19").values = [["保留版本"]];
summary.getRange("H19:L19").format.font = { name: font, size: 12, bold: true, color: dark };
summary.getRange("H20:I20").values = [["版本", "定位与结论"]];
styleHeader(summary.getRange("H20:I20"));
summary.getRange("H21:I24").values = [
  ["V2.7", "当前主基线。结构清楚，综合最稳定。"],
  ["V2.8", "目标排序专项。C05-10D/C20较强，C01不稳。"],
  ["V2.9", "可行性阶段专项。C05-30D较强，C05-10D/C22-10D退化。"],
  ["V2.5", "关键转折版本，但只有3-seed共同证据。"],
];
styleBody(summary.getRange("H21:I24"));
summary.getRange("H21:I21").format.fill = green;
summary.getRange("H22:I23").format.fill = blue;
summary.getRange("H24:I24").format.fill = amber;
summary.getRange("I21:I24").format.wrapText = true;
summary.getRange("H26:I27").values = [
  ["重要判断", "V2.7与V2.8平均rank仅差0.0125，不解释为整体胜负。"],
  ["统计限制", "5 runs不足以在双侧WSR下达到0.05。"],
];
summary.getRange("H26:I27").format.fill = light;
summary.getRange("H26:I27").format.font = { name: font, size: 10, bold: true, color: "#404040" };
summary.getRange("H26:I27").format.wrapText = false;
summary.getRange("H26:I27").format.rowHeight = 20;

summary.getRange("A31").values = [["共同3-seed开发轨迹"]];
summary.getRange("A31:F31").format.font = { name: font, size: 12, bold: true, color: dark };
summary.getRange("A32:F32").values = [["算法", "成功率", "平均运行rank", "胜/负DSI", "胜/负原SADE", "用途"]];
styleHeader(summary.getRange("A32:F32"));
const common3Rows = data.common_3seed_algorithm_summary.map((row) => [
  row.algorithm,
  Number(row.success_rate),
  Number(row.average_run_rank),
  `${row.wins_vs_dsi}/${row.losses_vs_dsi}`,
  `${row.wins_vs_sade}/${row.losses_vs_sade}`,
  row.algorithm === "sade_v2_5" ? "关键转折" : row.algorithm === "sade_v2_7" ? "当前主线" : "开发对照",
]);
summary.getRange(`A33:F${32 + common3Rows.length}`).values = common3Rows;
styleBody(summary.getRange(`A33:F${32 + common3Rows.length}`));
summary.getRange(`B33:B${32 + common3Rows.length}`).format.numberFormat = "0.0%";
summary.getRange(`C33:C${32 + common3Rows.length}`).format.numberFormat = "0.000";

summary.getRange("A44").values = [["历史25-run宽范围结果"]];
summary.getRange("A44:F44").format.font = { name: font, size: 12, bold: true, color: dark };
summary.getRange("A45:F48").values = [
  ["范围", "实例数", "DSI平均rank", "历史SADE平均rank", "SADE +/≈/−", "Holm调整p"],
  ["CEC2006", 13, 1.2308, 1.7692, "0/1/12", 0.0522],
  ["CEC2010", 12, 1.0000, 2.0000, "0/0/12", 0.0005],
  ["ALL", 43, 1.2558, 1.7442, "3/10/30", 0.0014],
];
styleHeader(summary.getRange("A45:F45"));
styleBody(summary.getRange("A46:F48"));
summary.getRange("C46:D48").format.numberFormat = "0.0000";
summary.getRange("F46:F48").format.numberFormat = "0.0000";
summary.getRange("A50:F51").values = [[
  "说明",
  "历史25-run使用后续源码修改前的原SADE，只作宽范围历史证据，不与当前5-run直接合并。",
  "", "", "", "",
], [
  "来源",
  "results/cec_10d30d_25runs_20260913_231757_699643/summary_statistics.csv",
  "", "", "", "",
]];
summary.getRange("A50:F51").format.font = { name: font, size: 9, color: "#666666", italic: true };

summary.getRange("A1:L55").format.verticalAlignment = "center";
summary.getRange("A:A").format.columnWidth = 17;
summary.getRange("B:F").format.columnWidth = 19;
summary.getRange("G:L").format.columnWidth = 14;
summary.getRange("H:H").format.columnWidth = 13;
summary.getRange("I:I").format.columnWidth = 48;

// Version history
styleTitle(versions, "A2", "SADE版本改动与实验结论", "F");
versions.getRange("A4:F4").values = [["版本", "父版本", "核心改动", "批次", "实验结论", "状态"]];
styleHeader(versions.getRange("A4:F4"));
const versionRows = data.version_changes.map((row) => [
  row.version, row.parent, row.main_change, Number(row.batch), row.development_conclusion, row.current_status,
]);
versions.getRange(`A5:F${4 + versionRows.length}`).values = versionRows;
styleBody(versions.getRange(`A5:F${4 + versionRows.length}`));
versions.getRange(`C5:C${4 + versionRows.length}`).format.wrapText = true;
versions.getRange(`E5:F${4 + versionRows.length}`).format.wrapText = true;
for (let i = 0; i < versionRows.length; i += 1) {
  const rowNumber = 5 + i;
  const status = versionRows[i][5];
  if (status === "retained_main_baseline") versions.getRange(`A${rowNumber}:F${rowNumber}`).format.fill = green;
  else if (status === "retained_specialist") versions.getRange(`A${rowNumber}:F${rowNumber}`).format.fill = blue;
  else if (status === "promising_limited_evidence") versions.getRange(`A${rowNumber}:F${rowNumber}`).format.fill = amber;
  else if (status === "failed_ablation") versions.getRange(`A${rowNumber}:F${rowNumber}`).format.fill = red;
}
versions.getRange("A:A").format.columnWidth = 12;
versions.getRange("B:B").format.columnWidth = 12;
versions.getRange("C:C").format.columnWidth = 58;
versions.getRange("D:D").format.columnWidth = 9;
versions.getRange("E:E").format.columnWidth = 48;
versions.getRange("F:F").format.columnWidth = 28;
versions.getRange(`A5:F${4 + versionRows.length}`).format.rowHeight = 42;
versions.freezePanes.freezeRows(4);

// Run data
const runHeaders = ["algorithm", "suite", "problem", "dimension", "seed", "objective", "total_violation", "feasible", "evaluations", "source"];
runs.getRange("A1:J1").values = [runHeaders];
styleHeader(runs.getRange("A1:J1"));
const runRows = data.selected_runs.map((row) => runHeaders.map((field) => row[field]));
runs.getRange(`A2:J${1 + runRows.length}`).values = runRows;
styleBody(runs.getRange(`A2:J${1 + runRows.length}`));
runs.getRange(`D2:E${1 + runRows.length}`).format.numberFormat = "0";
runs.getRange(`F2:G${1 + runRows.length}`).format.numberFormat = "0.000000E+00";
runs.getRange(`I2:I${1 + runRows.length}`).format.numberFormat = "0";
runs.getRange("A:A").format.columnWidth = 17;
runs.getRange("B:C").format.columnWidth = 12;
runs.getRange("D:E").format.columnWidth = 10;
runs.getRange("F:G").format.columnWidth = 18;
runs.getRange("H:I").format.columnWidth = 11;
runs.getRange("J:J").format.columnWidth = 72;
runs.freezePanes.freezeRows(1);
const runTable = runs.tables.add(`A1:J${1 + runRows.length}`, true, "RunDataTable");
runTable.style = "TableStyleMedium2";

// Historical 25-run results
const histHeaders = ["suite", "problem", "dimension", "metric", "dsi", "sade"];
historical.getRange("A1:F1").values = [histHeaders];
styleHeader(historical.getRange("A1:F1"));
const histRows = data.historical_25run_statistics.map((row) => histHeaders.map((field) => row[field]));
historical.getRange(`A2:F${1 + histRows.length}`).values = histRows;
styleBody(historical.getRange(`A2:F${1 + histRows.length}`));
historical.getRange("A:B").format.columnWidth = 16;
historical.getRange("C:C").format.columnWidth = 12;
historical.getRange("D:D").format.columnWidth = 34;
historical.getRange("E:F").format.columnWidth = 27;
historical.freezePanes.freezeRows(1);
const histTable = historical.tables.add(`A1:F${1 + histRows.length}`, true, "HistoricalStatisticsTable");
histTable.style = "TableStyleMedium2";

summary.tabColor = dark;
versions.tabColor = "#5B9BD5";
runs.tabColor = "#A5A5A5";
historical.tabColor = "#A5A5A5";

workbook.recalculate();
const summaryInspect = await workbook.inspect({
  kind: "table",
  sheetId: "Summary",
  range: "A10:L28",
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: 12,
});
console.log(summaryInspect.ndjson);
const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errorScan.ndjson);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(path.join(reportDir, "previews"), { recursive: true });
for (const [sheetName, range] of [
  ["Summary", "A1:L52"],
  ["Version history", "A1:F18"],
  ["Run data", "A1:J25"],
  ["Historical 25-run", "A1:F25"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  const safeName = sheetName.replaceAll(" ", "_");
  await fs.writeFile(path.join(reportDir, "previews", `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "SADE_experiment_review.xlsx");
await output.save(outputPath);
console.log(outputPath);
