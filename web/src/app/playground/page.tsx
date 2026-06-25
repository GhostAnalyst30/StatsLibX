"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Play,
  RotateCcw,
  Trash2,
  Download,
  Upload,
  FileCode,
  FileSpreadsheet,
  FolderTree,
  Search,
  Package,
  PackagePlus,
  Settings,
  Terminal,
  Bug,
  AlertTriangle,
  X,
  PanelLeftClose,
  PanelLeftOpen,
  PanelBottomClose,
  PanelBottomOpen,
  ChevronRight,
  ChevronDown,
  File,
  Loader2,
  Circle,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";

import { EditorView, keymap, lineNumbers, highlightActiveLine, dropCursor, rectangularSelection, crosshairCursor, highlightSpecialChars, drawSelection, highlightWhitespace } from "@codemirror/view";
import { EditorState } from "@codemirror/state";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";
import { closeBrackets, closeBracketsKeymap, autocompletion, completionKeymap } from "@codemirror/autocomplete";
import { indentOnInput, syntaxHighlighting, defaultHighlightStyle, bracketMatching, foldGutter } from "@codemirror/language";
import { history, historyKeymap, indentWithTab, defaultKeymap } from "@codemirror/commands";
import { highlightSelectionMatches, searchKeymap } from "@codemirror/search";

interface FileNode {
  name: string;
  icon: typeof FileCode | typeof FileSpreadsheet;
  language: string;
}

interface OutputLine {
  text: string;
  type: "stdout" | "stderr" | "system" | "error";
}

const FILES: Record<string, string> = {
  "main.py": `from statslibx import DescriptiveStats, InferentialStats
from statslibx.datasets import load_iris, generate_dataset

# Load sample data (returns pandas DataFrame)
iris = load_iris()
print(f"Iris dataset loaded! {len(iris)} samples")
print(f"Columns: {', '.join(iris.columns)}")

# Descriptive statistics
stats = DescriptiveStats(iris)
print("\\n=== Sepal Length Statistics ===")
print(f"Mean: {stats.mean('sepal_length'):.3f}")
print(f"Median: {stats.median('sepal_length'):.2f}")
print(f"Std Dev: {stats.std('sepal_length'):.3f}")

# Group stats by species
for species in iris["species"].unique():
    subset = iris[iris["species"] == species]
    gs = DescriptiveStats(subset)
    print(f"\\n{species}: mean={gs.mean('sepal_length'):.2f}, std={gs.std('sepal_length'):.2f}")

# Inferential statistics
infer = InferentialStats(iris)
result = infer.t_test_2sample("sepal_length", "sepal_width")
print("\\n=== T-Test: sepal_length vs sepal_width ===")
print(result)

# Synthetic data
schema = {"value": {"dist": "normal", "mean": 10, "std": 2, "type": "float"}}
synth = generate_dataset(n_rows=50, schema=schema)
print(f"\\nGenerated dataset shape: {synth.shape}")
`,
  "utils.py": `from statslibx import DescriptiveStats

def describe_summary(data, column=None, name="Data"):
    """Print a comprehensive summary of a dataset."""
    stats = DescriptiveStats(data)
    print(f"\\n{'='*50}")
    print(f"  {name} Summary")
    print(f"{'='*50}")
    if column:
        print(f"  Mean:       {stats.mean(column):.4f}")
        print(f"  Std Dev:    {stats.std(column):.4f}")
        print(f"  Median:     {stats.median(column):.4f}")
    else:
        print(f"  Mean:       {stats.mean():.4f}")
        print(f"  Std Dev:    {stats.std():.4f}")
        print(f"  Median:     {stats.median():.4f}")
    print(f"{'='*50}")
    return stats

if __name__ == "__main__":
    from statslibx.datasets import load_iris
    describe_summary(load_iris(), column="sepal_length", name="Iris Sepal Length")
`,
  "data.csv": `species,sepal_length,sepal_width,petal_length,petal_width
setosa,5.1,3.5,1.4,0.2
setosa,4.9,3.0,1.4,0.2
setosa,4.7,3.2,1.3,0.2
setosa,5.0,3.6,1.4,0.2
versicolor,6.0,2.9,4.5,1.5
versicolor,6.1,3.0,4.6,1.4
versicolor,5.8,2.7,4.1,1.0
versicolor,6.2,2.8,4.8,1.8
virginica,6.7,3.1,5.6,2.4
virginica,6.9,3.2,5.7,2.3
virginica,6.4,2.8,5.6,2.1
virginica,6.8,3.0,5.5,2.1
`,
  "test_analysis.py": `from statslibx import DescriptiveStats, InferentialStats, ComputationalStats
from statslibx.datasets import load_penguins, generate_dataset

# Generate synthetic dataset
schema = {
    "score": {"dist": "normal", "mean": 75, "std": 10, "type": "float"},
    "group": {"dist": "categorical", "choices": ["A", "B"]},
}
df = generate_dataset(n_rows=100, schema=schema, seed=42)
print(f"Generated {len(df)} rows")

stats = DescriptiveStats(df)
print(f"\\nScore mean: {stats.mean('score'):.2f}")
print(f"Score std:  {stats.std('score'):.2f}")

infer = InferentialStats(df)
result = infer.t_test_1sample("score", popmean=70)
print(f"\\nT-test vs 70: {result}")

cs = ComputationalStats(df)
print(f"\\nCorrelation matrix shape: {cs.correlation_analysis()['correlation_matrix'].shape}")

penguins = load_penguins()
print(f"\\nPenguins dataset: {len(penguins)} rows, columns: {list(penguins.columns)}")
`,
};

const FILE_ENTRIES: FileNode[] = [
  { name: "main.py", icon: FileCode, language: "python" },
  { name: "utils.py", icon: FileCode, language: "python" },
  { name: "data.csv", icon: FileSpreadsheet, language: "csv" },
  { name: "test_analysis.py", icon: FileCode, language: "python" },
];

const STUB_PYTHON_CODE = `
import sys
import math
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

class _TestResult:
    def __init__(self, test_name, statistic, pvalue):
        self.test_name = test_name
        self.statistic = statistic
        self.pvalue = pvalue
    def __repr__(self):
        sig = "significant" if self.pvalue < 0.05 else "not significant"
        return f"{self.test_name}: statistic={self.statistic:.4f}, p={self.pvalue:.4f} ({sig})"

class _DescriptiveStats:
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self._df = data
            self._series = None
        elif isinstance(data, pd.Series):
            self._df = None
            self._series = data
        else:
            self._series = pd.Series([float(x) for x in data])
            self._df = None

    def _values(self, column=None):
        if column is not None:
            return self._df[column].dropna()
        return self._series.dropna()

    def mean(self, column=None):
        return float(self._values(column).mean())
    def median(self, column=None):
        return float(self._values(column).median())
    def std(self, column=None, ddof=1):
        return float(self._values(column).std(ddof=ddof))
    def skewness(self, column=None):
        return float(scipy_stats.skew(self._values(column), bias=False))
    def summary(self):
        if self._df is not None:
            return type('S', (), {'to_dataframe': lambda self=None, format='wide': self._df.describe()})()
        s = self._series
        return {'count': len(s), 'mean': s.mean(), 'std': s.std()}

class _InferentialStats:
    def __init__(self, data):
        self._df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    def t_test_1sample(self, column, popmean=0, **kwargs):
        sample = self._df[column].dropna()
        t, p = scipy_stats.ttest_1samp(sample, popmean)
        return _TestResult(f"t-test 1-sample ({column})", float(t), float(p))

    def t_test_2sample(self, column1, column2, **kwargs):
        a = self._df[column1].dropna()
        b = self._df[column2].dropna()
        t, p = scipy_stats.ttest_ind(a, b, equal_var=kwargs.get('equal_var', True))
        return _TestResult(f"t-test 2-sample ({column1} vs {column2})", float(t), float(p))

    def confidence_interval(self, column, confidence=0.95):
        s = self._df[column].dropna()
        m = s.mean()
        se = s.std(ddof=1) / math.sqrt(len(s))
        h = se * scipy_stats.t.ppf((1 + confidence) / 2, len(s) - 1)
        return (m - h, m + h)

class _ComputationalStats:
    def __init__(self, data, seed=None):
        self._df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if seed is not None:
            np.random.seed(seed)

    def correlation_analysis(self, method='pearson'):
        return {'correlation_matrix': self._df.select_dtypes(include='number').corr(method=method)}

    def bootstrapping(self, column, n_samples=1000, statistic='mean', confidence_level=0.95, custom_func=None):
        col = self._df[column].dropna().values
        stats = []
        for _ in range(n_samples):
            sample = np.random.choice(col, size=len(col), replace=True)
            stats.append(np.mean(sample) if statistic == 'mean' else np.median(sample))
        lo = np.percentile(stats, (1 - confidence_level) / 2 * 100)
        hi = np.percentile(stats, (1 + confidence_level) / 2 * 100)
        return type('B', (), {'summary': lambda: {'ci': (lo, hi), 'original': float(np.mean(col))}})()

class _Preprocessing:
    def __init__(self, data):
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    def preview_data(self, n=5):
        return self.data.head(n)
    def describe_numeric(self):
        return self.data.select_dtypes(include='number').describe()

_IRIS = pd.DataFrame({
    'sepal_length': [5.1,4.9,4.7,5.0,5.4,6.0,6.1,5.8,6.2,6.7,6.9,6.4],
    'sepal_width': [3.5,3.0,3.2,3.6,3.9,2.9,3.0,2.7,2.8,3.1,3.2,2.8],
    'petal_length': [1.4,1.4,1.3,1.4,1.7,4.5,4.6,4.1,4.8,5.6,5.7,5.6],
    'petal_width': [0.2,0.2,0.2,0.2,0.4,1.5,1.4,1.0,1.8,2.4,2.3,2.1],
    'species': ['setosa']*5 + ['versicolor']*4 + ['virginica']*3,
})

_PENGUINS = pd.DataFrame({
    'species': ['Adelie','Adelie','Chinstrap','Gentoo'],
    'bill_length_mm': [39.1,39.5,46.5,50.1],
    'bill_depth_mm': [18.7,17.4,17.9,16.3],
    'flipper_length_mm': [181,186,192,230],
    'body_mass_g': [3750,3800,3500,5700],
})

def _generate_dataset(n_rows, schema, seed=42):
    rng = np.random.default_rng(seed)
    data = {}
    for col, cfg in schema.items():
        dist = cfg['dist']
        if dist == 'normal':
            vals = rng.normal(cfg.get('mean', 0), cfg.get('std', 1), n_rows)
        elif dist == 'categorical':
            vals = rng.choice(cfg['choices'], n_rows)
            data[col] = vals
            continue
        else:
            vals = rng.normal(0, 1, n_rows)
        if cfg.get('type') == 'int':
            vals = np.round(vals).astype(int)
        data[col] = vals
    return pd.DataFrame(data)

class _Datasets:
    load_iris = staticmethod(lambda backend='pandas': _IRIS.copy())
    load_penguins = staticmethod(lambda backend='pandas': _PENGUINS.copy())
    load_dataset = staticmethod(lambda name, backend='pandas', **kw: _IRIS.copy() if 'iris' in name else _PENGUINS.copy())
    generate_dataset = staticmethod(_generate_dataset)

stub_module = type(sys)('statslibx')
stub_module.DescriptiveStats = _DescriptiveStats
stub_module.InferentialStats = _InferentialStats
stub_module.ComputationalStats = _ComputationalStats
stub_module.Preprocessing = _Preprocessing
stub_module.datasets = _Datasets
stub_module.__version__ = '0.3.0-stub'
sys.modules['statslibx'] = stub_module
datasets_mod = type(sys)('statslibx.datasets')
datasets_mod.load_iris = _Datasets.load_iris
datasets_mod.load_penguins = _Datasets.load_penguins
datasets_mod.load_dataset = _Datasets.load_dataset
datasets_mod.generate_dataset = _Datasets.generate_dataset
sys.modules['statslibx.datasets'] = datasets_mod
`;

type SidePanel = "explorer" | "search" | "packages" | null;
type BottomTab = "output" | "terminal" | "problems";

export default function PlaygroundPage() {
  const [files, setFiles] = useState<Record<string, string>>({...FILES});
  const [activeFile, setActiveFile] = useState<string>("main.py");
  const [openFiles, setOpenFiles] = useState<string[]>(["main.py"]);
  const [output, setOutput] = useState<OutputLine[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [pyodideReady, setPyodideReady] = useState(false);
  const [pyodideLoading, setPyodideLoading] = useState(true);
  const [pyodideProgress, setPyodideProgress] = useState("");
  const [sidePanel, setSidePanel] = useState<SidePanel>("explorer");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [bottomPanelOpen, setBottomPanelOpen] = useState(true);
  const [bottomHeight, setBottomHeight] = useState(200);
  const [bottomTab, setBottomTab] = useState<BottomTab>("output");
  const [terminalInput, setTerminalInput] = useState("");
  const [terminalHistory, setTerminalHistory] = useState<string[]>([
    "Python REPL — enter Python expressions only (not shell commands)",
    "Example: from statslibx import DescriptiveStats; DescriptiveStats([1,2,3]).mean()",
  ]);
  const [problems, setProblems] = useState<{ file: string; line: number; msg: string }[]>([]);
  const [cursorPos, setCursorPos] = useState({ line: 1, col: 1 });
  const [execStatus, setExecStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [installedPackages, setInstalledPackages] = useState<{ name: string; installing: boolean; error?: string }[]>([
    { name: "numpy", installing: false },
    { name: "pandas", installing: false },
    { name: "scipy", installing: false },
    { name: "matplotlib", installing: false },
    { name: "statslibx", installing: false },
  ]);
  const [installInput, setInstallInput] = useState("");
  const [explorerCollapsed, setExplorerCollapsed] = useState<Record<string, boolean>>({});

  const editorRef = useRef<HTMLDivElement>(null);
  const editorViewRef = useRef<EditorView | null>(null);
  const pyodideRef = useRef<any>(null);
  const outputRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const terminalInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resizeRef = useRef<{ startY: number; startH: number } | null>(null);

  const addOutput = useCallback((line: OutputLine) => {
    setOutput(prev => [...prev, line]);
  }, []);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  useEffect(() => {
    let cancelled = false;

    async function initPyodide() {
      try {
        setPyodideProgress("Loading Pyodide runtime...");
        await loadScript("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

        if (cancelled) return;
        setPyodideProgress("Initializing Python environment...");

        const pyodide = await (window as any).loadPyodide({
          indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
        });

        if (cancelled) return;
        pyodideRef.current = pyodide;
        setPyodideProgress("Installing packages (numpy, pandas, scipy, matplotlib)...");

        await pyodide.loadPackage(["numpy", "pandas", "scipy", "matplotlib", "pytz"]);

        if (cancelled) return;
        setPyodideProgress("Injecting statslibx stubs...");

        pyodide.runPython(STUB_PYTHON_CODE);

        if (cancelled) return;
        setPyodideReady(true);
        setPyodideLoading(false);
        setPyodideProgress("");
        addOutput({ text: "Pyodide v0.26.4 ready — statslibx stubs loaded", type: "system" });
      } catch (err: any) {
        if (!cancelled) {
          setPyodideLoading(false);
          setPyodideProgress("Failed to load Pyodide");
          addOutput({ text: `Error: ${err.message || err}`, type: "error" });
        }
      }
    }

    initPyodide();
    return () => { cancelled = true; };
  }, [addOutput]);

  useEffect(() => {
    if (!editorRef.current || !pyodideReady) return;

    const startState = EditorState.create({
      doc: files[activeFile] || "",
      extensions: [
        lineNumbers(),
        highlightActiveLine(),
        dropCursor(),
        rectangularSelection(),
        crosshairCursor(),
        highlightSpecialChars(),
        drawSelection(),
        highlightWhitespace(),
        bracketMatching(),
        closeBrackets(),
        autocompletion(),
        foldGutter(),
        highlightSelectionMatches(),
        history(),
        python(),
        oneDark,
        indentOnInput(),
        syntaxHighlighting(defaultHighlightStyle),
        keymap.of([
          ...defaultKeymap,
          ...historyKeymap,
          ...closeBracketsKeymap,
          ...completionKeymap,
          ...searchKeymap,
          indentWithTab,
          { key: "Ctrl-Enter", run: () => { handleRunRef.current(); return true; } },
          { key: "Shift-Enter", run: () => { handleRunRef.current(); return true; } },
        ]),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) {
            const content = update.state.doc.toString();
            setFileContent(activeFile, content);
          }
          const pos = update.state.selection.main.head;
          const line = update.state.doc.lineAt(pos);
          setCursorPos({ line: line.number, col: pos - line.from + 1 });
        }),
        EditorView.theme({
          "&": { height: "100%" },
          ".cm-scroller": { overflow: "auto", fontFamily: "'DM Mono', monospace", fontSize: "13px" },
          ".cm-gutters": { borderRight: "1px solid rgba(255,255,255,0.06)" },
          ".cm-activeLineGutter": { backgroundColor: "rgba(124,106,247,0.1)" },
        }),
      ],
    });

    if (editorViewRef.current) {
      editorViewRef.current.destroy();
    }

    const view = new EditorView({
      state: startState,
      parent: editorRef.current,
    });

    editorViewRef.current = view;

    return () => {
      view.destroy();
      editorViewRef.current = null;
    };
  }, [activeFile, pyodideReady]);

  const setFileContent = useCallback((filename: string, content: string) => {
    setFiles(prev => ({ ...prev, [filename]: content }));
  }, []);

  const openFile = useCallback((name: string) => {
    setActiveFile(name);
    setOpenFiles(prev => prev.includes(name) ? prev : [...prev, name]);
  }, []);

  const closeFile = useCallback((name: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    setOpenFiles(prev => {
      const next = prev.filter(f => f !== name);
      if (next.length === 0) return prev;
      if (activeFile === name) {
        const idx = prev.indexOf(name);
        setActiveFile(next[Math.min(idx, next.length - 1)]);
      }
      return next;
    });
  }, [activeFile]);

  const handleRunRef = useRef<() => void>(() => {});

  const handleRun = useCallback(async () => {
    if (!pyodideRef.current || isRunning) return;

    const code = editorViewRef.current?.state.doc.toString() || files[activeFile] || "";

    setIsRunning(true);
    setExecStatus("running");
    setBottomPanelOpen(true);
    setBottomTab("output");
    addOutput({ text: `> python ${activeFile}`, type: "system" });

    const py = pyodideRef.current;

    try {
      const hasModule = await py.runPythonAsync(`import sys; 'statslibx' in sys.modules`);
      if (!hasModule) {
        await py.runPythonAsync(STUB_PYTHON_CODE);
      }

      py.setStdout?.({
        batched: (msg: string) => addOutput({ text: msg, type: "stdout" }),
      });
      py.setStderr?.({
        batched: (msg: string) => {
          addOutput({ text: msg, type: "stderr" });
          setProblems(prev => [...prev, { file: activeFile, line: 1, msg }]);
        },
      });

      const result = await py.runPythonAsync(code);

      if (result !== undefined && result !== null) {
        addOutput({ text: String(result), type: "stdout" });
      }

      setExecStatus("done");
      addOutput({ text: `\n[Done] exited with code 0`, type: "system" });
    } catch (err: any) {
      const msg = err.message || String(err);
      addOutput({ text: `\nTraceback (most recent call last):`, type: "stderr" });
      addOutput({ text: msg, type: "error" });
      setProblems(prev => [...prev, { file: activeFile, line: 1, msg }]);
      setExecStatus("error");
      addOutput({ text: `[Done] exited with code 1`, type: "system" });
    }

    setIsRunning(false);
  }, [isRunning, activeFile, files, addOutput]);

  handleRunRef.current = handleRun;

  const handleClearOutput = useCallback(() => {
    setOutput([]);
    setProblems([]);
  }, []);

  const handleReset = useCallback(async () => {
    setOutput([]);
    setProblems([]);
    setExecStatus("idle");
    addOutput({ text: "Resetting Python session...", type: "system" });

    if (pyodideRef.current) {
      try {
        await pyodideRef.current.runPythonAsync(STUB_PYTHON_CODE);
        addOutput({ text: "Session reset complete. All variables cleared.", type: "system" });
      } catch {
        addOutput({ text: "Session reset.", type: "system" });
      }
    }
  }, [addOutput]);

  const handleDownload = useCallback(() => {
    const code = editorViewRef.current?.state.doc.toString() || files[activeFile] || "";
    const blob = new Blob([code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = activeFile;
    a.click();
    URL.revokeObjectURL(url);
  }, [activeFile, files]);

  const handleTerminalSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const input = terminalInputRef.current;
    if (!input) return;
    const cmd = input.value.trim();
    if (!cmd) return;
    input.value = "";
    setTerminalInput("");

    if (!pyodideRef.current) {
      setTerminalHistory(prev => [...prev, `>>> ${cmd}`, "  Pyodide not ready yet."]);
      return;
    }

    setTerminalHistory(prev => [...prev, `>>> ${cmd}`]);

    const py = pyodideRef.current;
    try {
      const hasModule = await py.runPythonAsync(`import sys; 'statslibx' in sys.modules`);
      if (!hasModule) {
        await py.runPythonAsync(STUB_PYTHON_CODE);
      }

      py.setStdout?.({ batched: (msg: string) => {
        setTerminalHistory(prev => [...prev, msg]);
      }});
      py.setStderr?.({ batched: (msg: string) => {
        setTerminalHistory(prev => [...prev, `Error: ${msg}`]);
      }});

      const result = await py.runPythonAsync(cmd);
      if (result !== undefined && result !== null) {
        setTerminalHistory(prev => [...prev, String(result)]);
      }
    } catch (err: any) {
      setTerminalHistory(prev => [...prev, `Error: ${err.message || err}`]);
    }
  }, []);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const name = file.name;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const content = ev.target?.result as string;
      setFiles(prev => ({ ...prev, [name]: content }));
      setUploadedFiles(prev => prev.includes(name) ? prev : [...prev, name]);
      setActiveFile(name);
      setOpenFiles(prev => prev.includes(name) ? prev : [...prev, name]);
    };
    reader.readAsText(file);
    e.target.value = "";
  }, []);

  const handleDeleteFile = useCallback((name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setUploadedFiles(prev => prev.filter(f => f !== name));
    setFiles(prev => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
    if (activeFile === name) {
      const remaining = openFiles.filter(f => f !== name);
      setActiveFile(remaining.length > 0 ? remaining[remaining.length - 1] : "main.py");
    }
    setOpenFiles(prev => prev.filter(f => f !== name));
  }, [activeFile, openFiles]);

  const handleInstallPackage = useCallback(async (pkgName: string) => {
    if (!pyodideRef.current || !pkgName.trim()) return;

    setInstalledPackages(prev => prev.some(p => p.name === pkgName)
      ? prev.map(p => p.name === pkgName ? { ...p, installing: true, error: undefined } : p)
      : [...prev, { name: pkgName, installing: true }]
    );

    try {
      const py = pyodideRef.current;
      await py.runPythonAsync(`
import micropip
await micropip.install('${pkgName.replace(/'/g, "\\'")}')
`);
      setInstalledPackages(prev => prev.map(p =>
        p.name === pkgName ? { ...p, installing: false, error: undefined } : p
      ));
      addOutput({ text: `Package installed: ${pkgName}`, type: "system" });
    } catch (err: any) {
      setInstalledPackages(prev => prev.map(p =>
        p.name === pkgName ? { ...p, installing: false, error: err.message || "Installation failed" } : p
      ));
      addOutput({ text: `Failed to install ${pkgName}: ${err.message || err}`, type: "error" });
    }
  }, [addOutput]);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    resizeRef.current = { startY: e.clientY, startH: bottomHeight };
    const onMove = (ev: MouseEvent) => {
      if (!resizeRef.current) return;
      const delta = resizeRef.current.startY - ev.clientY;
      const newH = Math.min(Math.max(resizeRef.current.startH - delta, 100), 500);
      setBottomHeight(newH);
    };
    const onUp = () => {
      resizeRef.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [bottomHeight]);

  function getFileIcon(name: string) {
    if (name.endsWith(".csv")) return FileSpreadsheet;
    return FileCode;
  }

  const explorerFiles = useMemo(() => {
    const builtIn = FILE_ENTRIES;
    const uploaded = uploadedFiles.map(name => ({
      name,
      icon: name.endsWith(".csv") ? FileSpreadsheet as typeof FileCode : FileCode as typeof FileCode,
      language: name.endsWith(".py") ? "python" : "text",
    }));
    return [...builtIn, ...uploaded];
  }, [uploadedFiles]);

  const statusColor = execStatus === "idle" ? "text-muted"
    : execStatus === "running" ? "text-accent2"
    : execStatus === "done" ? "text-green-400"
    : "text-red-400";

  const statusText = execStatus === "idle" ? "Idle"
    : execStatus === "running" ? "Running..."
    : execStatus === "done" ? "Done"
    : "Error";

  return (
    <div className="h-[calc(100vh-3.5rem)] w-full bg-[#09090f] flex flex-col overflow-hidden select-none">
      <div className="flex-1 flex min-h-0">
        {/* Activity Bar */}
        <div className="w-[50px] flex-shrink-0 bg-[#111118] border-r border-[rgba(255,255,255,0.06)] flex flex-col items-center py-2 gap-1 z-10">
          <SidebarButton
            icon={FolderTree}
            active={sidePanel === "explorer"}
            tooltip="File Explorer"
            onClick={() => {
              if (sidePanel === "explorer") { setSidebarOpen(!sidebarOpen); } else { setSidePanel("explorer"); setSidebarOpen(true); }
            }}
          />
          <SidebarButton
            icon={Search}
            active={sidePanel === "search"}
            tooltip="Search"
            onClick={() => {
              if (sidePanel === "search") { setSidebarOpen(!sidebarOpen); } else { setSidePanel("search"); setSidebarOpen(true); }
            }}
          />
          <SidebarButton
            icon={Package}
            active={sidePanel === "packages"}
            tooltip="Python Packages"
            onClick={() => {
              if (sidePanel === "packages") { setSidebarOpen(!sidebarOpen); } else { setSidePanel("packages"); setSidebarOpen(true); }
            }}
          />
          <div className="flex-1" />
          <SidebarButton
            icon={Settings}
            active={false}
            tooltip="Settings"
            onClick={() => {}}
          />
        </div>

        {/* Sidebar */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.div
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 250, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.15, ease: "easeInOut" }}
              className="bg-[#111118] border-r border-[rgba(255,255,255,0.06)] overflow-hidden flex-shrink-0"
              style={{ minWidth: sidebarOpen ? 250 : 0 }}
            >
              <div className="w-[250px] h-full flex flex-col">
                {/* Sidebar Header */}
                <div className="h-9 flex items-center px-4 text-[11px] font-semibold text-muted uppercase tracking-wider border-b border-[rgba(255,255,255,0.06)]">
                  {sidePanel === "explorer" && "EXPLORER"}
                  {sidePanel === "search" && "SEARCH"}
                  {sidePanel === "packages" && "PACKAGES"}
                </div>

                {/* Explorer View */}
                {sidePanel === "explorer" && (
                  <div className="flex-1 overflow-y-auto py-1">
                    {explorerFiles.map((file) => {
                      const Icon = file.icon;
                      const isActive = activeFile === file.name;
                      const isUploaded = uploadedFiles.includes(file.name);
                      return (
                        <div
                          key={file.name}
                          onClick={() => openFile(file.name)}
                          className={`w-full flex items-center gap-2 px-3 py-1.5 text-sm transition-colors cursor-pointer group ${
                            isActive
                              ? "bg-[rgba(124,106,247,0.12)] text-white border-l-2 border-accent"
                              : "text-muted hover:text-white hover:bg-white/[0.04] border-l-2 border-transparent"
                          }`}
                        >
                          <Icon className="w-4 h-4 flex-shrink-0" style={{ color: file.name.endsWith(".csv") ? "#4fd1c5" : "#7c6af7" }} />
                          <span className="truncate flex-1">{file.name}</span>
                          {isUploaded && (
                            <button
                              onClick={(e) => handleDeleteFile(file.name, e)}
                              className="p-0.5 rounded hover:bg-red-500/20 text-muted hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100 cursor-pointer flex-shrink-0"
                              title="Remove file"
                            >
                              <X className="w-3 h-3" />
                            </button>
                          )}
                        </div>
                      );
                    })}
                    <div className="px-3 pt-3">
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".py,.csv,.txt,.json,.md,.js,.ts,.ipynb"
                        onChange={handleFileUpload}
                        className="hidden"
                      />
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-muted hover:text-white hover:bg-white/[0.04] rounded transition-colors cursor-pointer border border-dashed border-[rgba(255,255,255,0.1)]"
                      >
                        <Upload className="w-3.5 h-3.5" />
                        Upload File
                      </button>
                    </div>
                  </div>
                )}

                {/* Search View */}
                {sidePanel === "search" && (
                  <div className="flex-1 p-3">
                    <input
                      type="text"
                      placeholder="Search files..."
                      className="w-full bg-[#09090f] border border-[rgba(255,255,255,0.08)] rounded px-3 py-1.5 text-sm text-white placeholder-muted outline-none focus:border-accent/50 transition-colors"
                    />
                    <p className="text-xs text-muted mt-3">Search across files in your workspace.</p>
                  </div>
                )}

                {/* Packages View */}
                {sidePanel === "packages" && (
                  <div className="flex-1 p-3 space-y-2 overflow-y-auto flex flex-col">
                    <div className="text-xs font-medium text-muted uppercase tracking-wider mb-1">Installed Packages</div>
                    <div className="flex-1 space-y-1">
                      {installedPackages.map((pkg) => (
                        <div key={pkg.name} className="flex items-center gap-2 text-sm text-white/80">
                          {pkg.installing ? (
                            <Loader2 className="w-3.5 h-3.5 text-accent animate-spin flex-shrink-0" />
                          ) : pkg.error ? (
                            <AlertTriangle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                          ) : (
                            <CheckCircle2 className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
                          )}
                          <span className="truncate">{pkg.name}</span>
                          {pkg.error && <span className="text-[10px] text-red-400 truncate ml-auto">{pkg.error}</span>}
                        </div>
                      ))}
                    </div>
                    <form
                      onSubmit={(e) => {
                        e.preventDefault();
                        const name = installInput.trim();
                        if (!name || !pyodideReady) return;
                        handleInstallPackage(name);
                        setInstallInput("");
                      }}
                      className="flex items-center gap-2 pt-2 border-t border-[rgba(255,255,255,0.06)]"
                    >
                      <input
                        type="text"
                        value={installInput}
                        onChange={(e) => setInstallInput(e.target.value)}
                        placeholder="Package name..."
                        disabled={!pyodideReady}
                        className="flex-1 bg-[#09090f] border border-[rgba(255,255,255,0.08)] rounded px-2.5 py-1.5 text-xs font-mono text-white placeholder-muted outline-none focus:border-accent/50 transition-colors disabled:opacity-40"
                      />
                      <button
                        type="submit"
                        disabled={!pyodideReady || !installInput.trim()}
                        className="flex items-center gap-1 px-2.5 py-1.5 text-xs rounded bg-accent/20 text-accent hover:bg-accent/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer flex-shrink-0"
                      >
                        <PackagePlus className="w-3.5 h-3.5" />
                        Install
                      </button>
                    </form>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Content */}
        <div className="flex-1 flex flex-col min-w-0 bg-[#09090f]">
          {/* Editor Toolbar */}
          <div className="h-10 flex items-center justify-between px-3 bg-[#111118] border-b border-[rgba(255,255,255,0.06)] gap-2">
            <div className="flex items-center gap-1">
              <button
                onClick={handleRun}
                disabled={!pyodideReady || isRunning}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors cursor-pointer"
              >
                {isRunning ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Play className="w-3.5 h-3.5 fill-current" />
                )}
                Run
              </button>
              <button
                onClick={handleClearOutput}
                className="flex items-center gap-1 px-2.5 py-1.5 text-xs text-muted hover:text-white rounded hover:bg-white/[0.06] transition-colors cursor-pointer"
                title="Clear Output"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={handleReset}
                className="flex items-center gap-1 px-2.5 py-1.5 text-xs text-muted hover:text-white rounded hover:bg-white/[0.06] transition-colors cursor-pointer"
                title="Reset Session"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={handleDownload}
                className="flex items-center gap-1 px-2.5 py-1.5 text-xs text-muted hover:text-white rounded hover:bg-white/[0.06] transition-colors cursor-pointer"
                title="Download Code"
              >
                <Download className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-[10px] font-mono text-muted bg-white/[0.04] px-2 py-0.5 rounded">Pyodide v0.26.4</span>
            </div>
          </div>

          {/* Tabs */}
          <div className="h-9 flex items-center bg-[#0c0c14] border-b border-[rgba(255,255,255,0.06)] overflow-x-auto">
            {openFiles.map((name) => {
              const Icon = getFileIcon(name);
              const isActive = name === activeFile;
              return (
                <div
                  key={name}
                  onClick={() => setActiveFile(name)}
                  className={`flex items-center gap-1.5 px-3 h-full text-xs border-r border-[rgba(255,255,255,0.06)] cursor-pointer transition-colors whitespace-nowrap ${
                    isActive
                      ? "bg-[#09090f] text-white border-t-2 border-t-accent"
                      : "bg-transparent text-muted hover:text-white hover:bg-white/[0.03] border-t-2 border-t-transparent"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" style={{ color: name.endsWith(".csv") ? "#4fd1c5" : "#7c6af7" }} />
                  <span>{name}</span>
                  <button
                    onClick={(e) => closeFile(name, e)}
                    className="ml-1 p-0.5 rounded hover:bg-white/[0.1] text-muted hover:text-white transition-colors cursor-pointer"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              );
            })}
          </div>

          {/* Editor */}
          <div className="flex-1 relative overflow-hidden">
            {!pyodideReady && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#09090f] z-20 gap-3">
                <Loader2 className="w-8 h-8 text-accent animate-spin" />
                <p className="text-sm text-muted font-mono">{pyodideProgress || "Initializing..."}</p>
                <div className="w-48 h-1 bg-[#1a1a28] rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-accent rounded-full"
                    animate={{ x: ["-100%", "100%"] }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                  />
                </div>
              </div>
            )}
            <div ref={editorRef} className="h-full" />
          </div>

          {/* Resize Handle */}
          <div
            onMouseDown={handleResizeStart}
            className="h-1.5 bg-[#111118] hover:bg-accent/30 cursor-row-resize transition-colors flex-shrink-0 relative z-10"
          >
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-8 h-[2px] rounded-full bg-white/[0.08]" />
            </div>
          </div>

          {/* Bottom Panel */}
          <AnimatePresence>
            {bottomPanelOpen && (
              <motion.div
                initial={{ height: 0 }}
                animate={{ height: bottomHeight }}
                exit={{ height: 0 }}
                transition={{ duration: 0.15, ease: "easeInOut" }}
                className="bg-[#111118] border-t border-[rgba(255,255,255,0.06)] flex-shrink-0 overflow-hidden"
                style={{ minHeight: bottomPanelOpen ? bottomHeight : 0 }}
              >
                <div className="h-full flex flex-col">
                  {/* Panel Tabs */}
                  <div className="h-8 flex items-center bg-[#0c0c14] border-b border-[rgba(255,255,255,0.06)]">
                    <PanelTab
                      label="Output"
                      icon={Terminal}
                      active={bottomTab === "output"}
                      onClick={() => setBottomTab("output")}
                    />
                    <PanelTab
                      label="Python REPL"
                      icon={Terminal}
                      active={bottomTab === "terminal"}
                      onClick={() => setBottomTab("terminal")}
                    />
                    <PanelTab
                      label="Problems"
                      icon={AlertTriangle}
                      active={bottomTab === "problems"}
                      onClick={() => setBottomTab("problems")}
                      count={problems.length}
                    />
                    <div className="flex-1" />
                    <button
                      onClick={() => setBottomPanelOpen(false)}
                      className="px-2 h-full text-muted hover:text-white hover:bg-white/[0.05] transition-colors cursor-pointer"
                    >
                      <PanelBottomClose className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Panel Content */}
                  <div className="flex-1 overflow-hidden">
                    {bottomTab === "output" && (
                      <div ref={outputRef} className="h-full overflow-y-auto p-3 font-mono text-xs leading-relaxed">
                        {output.length === 0 ? (
                          <span className="text-muted italic">No output yet. Click Run to execute code.</span>
                        ) : (
                          output.map((line, i) => (
                            <div
                              key={i}
                              className={`whitespace-pre-wrap break-all ${
                                line.type === "stdout" ? "text-gray-300"
                                : line.type === "stderr" ? "text-yellow-300"
                                : line.type === "error" ? "text-red-400"
                                : "text-muted"
                              }`}
                            >
                              {line.text}
                            </div>
                          ))
                        )}
                      </div>
                    )}

                    {bottomTab === "terminal" && (
                      <div className="h-full flex flex-col">
                        <div className="flex-1 overflow-y-auto p-3 font-mono text-xs leading-relaxed">
                          {terminalHistory.map((line, i) => (
                            <div key={i} className={line.startsWith("$") ? "text-accent2" : "text-muted"}>
                              {line}
                            </div>
                          ))}
                        </div>
                        <form onSubmit={handleTerminalSubmit} className="flex items-center gap-2 p-2 border-t border-[rgba(255,255,255,0.06)]">
                          <span className="text-accent2 font-mono text-xs">$</span>
                          <input
                            ref={terminalInputRef}
                            type="text"
                            value={terminalInput}
                            onChange={(e) => setTerminalInput(e.target.value)}
                            className="flex-1 bg-transparent border-none outline-none text-xs font-mono text-white placeholder-muted"
                            placeholder="Type a command..."
                          />
                        </form>
                      </div>
                    )}

                    {bottomTab === "problems" && (
                      <div className="h-full overflow-y-auto p-3 font-mono text-xs">
                        {problems.length === 0 ? (
                          <span className="text-muted italic">No problems detected.</span>
                        ) : (
                          problems.map((prob, i) => (
                            <div key={i} className="flex items-start gap-2 py-1.5 text-red-400 border-b border-[rgba(255,255,255,0.04)]">
                              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                              <div>
                                <span className="text-muted">{prob.file}:{prob.line}</span> {prob.msg}
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Status Bar */}
      <div className="h-[22px] flex items-center justify-between px-3 bg-[#1a1a28] border-t border-[rgba(255,255,255,0.06)] text-[11px] text-muted font-mono flex-shrink-0">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <Circle className={`w-2.5 h-2.5 ${statusColor} ${execStatus === "running" ? "animate-pulse" : ""}`} />
            {statusText}
          </span>
          <span>Python 3.12 · Pyodide v0.26.4</span>
          {!pyodideReady && <span className="text-yellow-400">(loading...)</span>}
        </div>
        <div className="flex items-center gap-4">
          <span>Ln {cursorPos.line}, Col {cursorPos.col}</span>
          <span>Spaces: 4</span>
          <span>UTF-8</span>
          <span className="flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${pyodideReady ? "bg-green-400" : "bg-yellow-400"}`} />
            {pyodideReady ? "Ready" : "Booting"}
          </span>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-0.5 hover:text-white transition-colors cursor-pointer"
            title="Toggle Sidebar"
          >
            {sidebarOpen ? <PanelLeftClose className="w-3 h-3" /> : <PanelLeftOpen className="w-3 h-3" />}
          </button>
          <button
            onClick={() => setBottomPanelOpen(!bottomPanelOpen)}
            className="p-0.5 hover:text-white transition-colors cursor-pointer"
            title="Toggle Panel"
          >
            {bottomPanelOpen ? <PanelBottomClose className="w-3 h-3" /> : <PanelBottomOpen className="w-3 h-3" />}
          </button>
        </div>
      </div>
    </div>
  );
}

function SidebarButton({
  icon: Icon,
  active,
  tooltip,
  onClick,
}: {
  icon: typeof FolderTree;
  active: boolean;
  tooltip: string;
  onClick: () => void;
}) {
  return (
    <div className="relative group">
      <button
        onClick={onClick}
        className={`flex items-center justify-center w-10 h-10 rounded-lg transition-colors cursor-pointer ${
          active
            ? "text-accent bg-accent/10 border-l-2 border-accent rounded-none"
            : "text-muted hover:text-white hover:bg-white/[0.05]"
        }`}
      >
        <Icon className="w-5 h-5" />
      </button>
      <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 bg-[#1e1e2e] text-[11px] text-white rounded shadow-lg whitespace-nowrap opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-150 z-50 pointer-events-none border border-[rgba(255,255,255,0.06)]">
        {tooltip}
      </div>
    </div>
  );
}

function PanelTab({
  label,
  icon: Icon,
  active,
  onClick,
  count,
}: {
  label: string;
  icon: typeof Terminal;
  active: boolean;
  onClick: () => void;
  count?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 h-full text-[11px] font-medium transition-colors cursor-pointer border-r border-[rgba(255,255,255,0.06)] ${
        active
          ? "text-white bg-[#111118] border-t-2 border-t-accent"
          : "text-muted hover:text-white hover:bg-white/[0.03] border-t-2 border-t-transparent"
      }`}
    >
      <Icon className="w-3.5 h-3.5" />
      {label}
      {count !== undefined && count > 0 && (
        <span className="bg-red-500/20 text-red-400 text-[10px] px-1.5 rounded-full">{count}</span>
      )}
    </button>
  );
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}
