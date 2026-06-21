import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, FileText, RefreshCw, Copy, Edit, Check, 
  Cpu, Eye, EyeOff, LayoutGrid, HelpCircle, AlertTriangle, 
  Sparkles, CheckCircle, ListCollapse, BookOpen, Download, FileCode, Network,
  ScanLine, Trash2
} from 'lucide-react';

// API Base URL
const API_URL = "http://localhost:8000";

// KaTeX LaTeX Renderer Component (Enlarged)
const LaTeXRenderer = ({ latex }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current && window.katex) {
      try {
        const cleanLatex = latex
          .replace(/\\ /g, ' ')
          .trim();
        
        window.katex.render(cleanLatex || '\\text{Rỗng}', containerRef.current, {
          displayMode: true,
          throwOnError: false,
          macros: {
            "\\color": "\\textcolor"
          }
        });
      } catch (err) {
        containerRef.current.innerHTML = `<span class="text-danger font-monospace fs-5">${err.message}</span>`;
      }
    }
  }, [latex]);

  return <div ref={containerRef} className="latex-render-container my-3 text-center bg-light p-3 rounded border border-secondary border-opacity-25" />;
};

// KaTeX Inline LaTeX Renderer Component
const LaTeXInlineRenderer = ({ latex }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current && window.katex) {
      try {
        window.katex.render(latex.trim(), containerRef.current, {
          displayMode: false,
          throwOnError: false,
        });
      } catch (err) {
        containerRef.current.innerHTML = `<span class="text-danger font-monospace small">${err.message}</span>`;
      }
    }
  }, [latex]);

  return <span ref={containerRef} className="mx-1 font-semibold text-purple" />;
};

// Parses bold **text** and inline code `code`
const parseTextFormatting = (text, keyPrefix) => {
  if (!text) return "";
  
  const boldParts = text.split('**');
  return boldParts.map((bPart, bIdx) => {
    const isBold = bIdx % 2 === 1;
    
    const codeParts = bPart.split('`');
    const content = codeParts.map((cPart, cIdx) => {
      if (cIdx % 2 === 1) {
        return (
          <code key={`code-${keyPrefix}-${bIdx}-${cIdx}`} className="font-monospace bg-light px-1.5 py-0.5 rounded text-pink border border-secondary border-opacity-35" style={{ fontSize: '13px' }}>
            {cPart}
          </code>
        );
      }
      return cPart;
    });

    if (isBold) {
      return <strong key={`bold-${keyPrefix}-${bIdx}`} className="fw-bold text-dark">{content}</strong>;
    }
    return <span key={`span-${keyPrefix}-${bIdx}`}>{content}</span>;
  });
};

// Custom Markdown Renderer with LaTeX math rendering support
const MarkdownMathRenderer = ({ content }) => {
  if (!content) {
    return (
      <div className="text-secondary italic p-4 text-center">
        Chưa có kết quả quét trang. Hãy tải ảnh lên và chạy phân tích ở cột bên trái.
      </div>
    );
  }

  const lines = content.split('\n');
  const renderedElements = [];
  let currentList = [];
  
  const flushList = (key) => {
    if (currentList.length > 0) {
      renderedElements.push(
        <ul key={`ul-${key}`} className="ps-4 mb-3" style={{ listStyleType: 'disc' }}>
          {currentList}
        </ul>
      );
      currentList = [];
    }
  };

  const renderLineWithMath = (text) => {
    if (!text) return "";
    
    const elements = [];
    let lastIndex = 0;
    let match;
    let matchKey = 0;
    
    // Match $$...$$ (display math, group 1) or $...$ (inline math, group 2)
    const mathRegex = /\$\$(.*?)\$\$|\$(.*?)\$/g;

    while ((match = mathRegex.exec(text)) !== null) {
      const matchIndex = match.index;
      const displayMath = match[1];
      const inlineMath = match[2];

      if (matchIndex > lastIndex) {
        const precedingText = text.substring(lastIndex, matchIndex);
        const formatted = parseTextFormatting(precedingText, `text-${matchKey}`);
        if (formatted) {
          elements.push(...(Array.isArray(formatted) ? formatted : [formatted]));
        }
      }

      if (displayMath !== undefined) {
        elements.push(
          <LaTeXRenderer key={`display-math-${matchKey}`} latex={displayMath} />
        );
      } else if (inlineMath !== undefined) {
        elements.push(
          <LaTeXInlineRenderer key={`inline-math-${matchKey}`} latex={inlineMath} />
        );
      }

      lastIndex = mathRegex.lastIndex;
      matchKey++;
    }

    if (lastIndex < text.length) {
      const trailingText = text.substring(lastIndex);
      const formatted = parseTextFormatting(trailingText, `text-end`);
      if (formatted) {
        elements.push(...(Array.isArray(formatted) ? formatted : [formatted]));
      }
    }

    return elements.length > 0 ? elements : parseTextFormatting(text, 'all');
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmedLine = line.trim();
    
    if (trimmedLine.startsWith('- ') || trimmedLine.startsWith('* ')) {
      const marker = trimmedLine.startsWith('- ') ? '-' : '*';
      const itemText = line.substring(line.indexOf(marker) + 2);
      currentList.push(
        <li key={`li-${i}`} className="mb-1 text-dark">
          {renderLineWithMath(itemText)}
        </li>
      );
      continue;
    }
    
    flushList(i);
    
    if (trimmedLine.startsWith('# ')) {
      renderedElements.push(
        <h1 key={`h1-${i}`} className="h3 fw-bold text-primary mt-4 mb-3 border-bottom border-secondary border-opacity-35 pb-2">
          {renderLineWithMath(line.substring(line.indexOf('#') + 2))}
        </h1>
      );
    } else if (trimmedLine.startsWith('## ')) {
      renderedElements.push(
        <h2 key={`h2-${i}`} className="h4 fw-bold text-purple mt-3 mb-2">
          {renderLineWithMath(line.substring(line.indexOf('#') + 3))}
        </h2>
      );
    } else if (trimmedLine.startsWith('### ')) {
      renderedElements.push(
        <h3 key={`h3-${i}`} className="h5 fw-bold text-dark mt-3 mb-2">
          {renderLineWithMath(line.substring(line.indexOf('#') + 4))}
        </h3>
      );
    } else if (
      trimmedLine.startsWith('$$') && 
      trimmedLine.endsWith('$$') && 
      trimmedLine.indexOf('$$', 2) === trimmedLine.length - 2 && 
      trimmedLine.length >= 4
    ) {
      const formula = trimmedLine.substring(2, trimmedLine.length - 2).trim();
      renderedElements.push(
        <LaTeXRenderer key={`block-math-${i}`} latex={formula} />
      );
    } else if (trimmedLine === '$$') {
      let formulaLines = [];
      let foundEnd = false;
      let j = i + 1;
      for (; j < lines.length; j++) {
        if (lines[j].trim() === '$$') {
          foundEnd = true;
          break;
        }
        formulaLines.push(lines[j]);
      }
      if (foundEnd) {
        renderedElements.push(
          <LaTeXRenderer key={`block-math-${i}`} latex={formulaLines.join('\n')} />
        );
        i = j;
      } else {
        renderedElements.push(<p key={`p-${i}`} className="mb-2 text-dark">{renderLineWithMath(line)}</p>);
      }
    } else if (line.trim() === '') {
      renderedElements.push(<div key={`empty-${i}`} className="mb-2" style={{ height: '8px' }}></div>);
    } else {
      renderedElements.push(
        <p key={`p-${i}`} className="mb-2 text-dark-emphasis leading-relaxed" style={{ fontSize: '15px' }}>
          {renderLineWithMath(line)}
        </p>
      );
    }
  }
  
  flushList('final');
  
  return <div className="markdown-body text-start">{renderedElements}</div>;
};

export default function App() {
  // Document State
  const [recognitionEngine, setRecognitionEngine] = useState("pix2text");
  const [resultEngine, setResultEngine] = useState("pix2text");
  const [selectedFile, setSelectedFile] = useState(null);
  const [docId, setDocId] = useState(null);
  const [filename, setFilename] = useState("");
  const [elements, setElements] = useState([]);
  const [recognitionBoxes, setRecognitionBoxes] = useState([]);
  const [markdown, setMarkdown] = useState("");
  const [annotatedImg, setAnnotatedImg] = useState("");
  const [originalImg, setOriginalImg] = useState("");
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [isRegionMode, setIsRegionMode] = useState(false);
  const [customBox, setCustomBox] = useState(null);
  const [drawingBox, setDrawingBox] = useState(null);
  const [regionScanning, setRegionScanning] = useState(false);
  
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [activeTab, setActiveTab] = useState("elements"); // elements, markdown
  const [rightPanelTab, setRightPanelTab] = useState("visual"); // visual, source
  
  const [loading, setLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [systemHealth, setSystemHealth] = useState(null);

  // States to handle inline editing on elements text
  const [editingTexts, setEditingTexts] = useState({});
  const [editingIndex, setEditingIndex] = useState(null);

  // Copy state variables
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [markdownCopied, setMarkdownCopied] = useState(false);

  const imageRef = useRef(null);

  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setSystemHealth(data);
      }
    } catch (e) {
      console.error("Backend health check failed:", e);
      setSystemHealth({ status: "offline" });
    }
  };

  const handleFileSelection = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const fileExt = file.name.split('.').pop().toLowerCase();
    if (!['png', 'jpg', 'jpeg', 'webp'].includes(fileExt)) {
      setUploadError("Chỉ hỗ trợ tải lên tệp ảnh định dạng (.png, .jpg, .jpeg, .webp).");
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setFilename(file.name);
    setOriginalImg(URL.createObjectURL(file));
    setUploadError("");
  };

  const handleImageProcessing = async () => {
    const file = selectedFile;
    if (!file) {
      setUploadError("Vui lòng chọn một tệp ảnh trước khi xử lý.");
      return;
    }

    setIsUploading(true);
    setUploadError("");
    setElements([]);
    setRecognitionBoxes([]);
    setMarkdown("");
    setAnnotatedImg("");
    setImageSize({ width: 0, height: 0 });
    setIsRegionMode(false);
    setCustomBox(null);
    setDrawingBox(null);
    setSelectedIdx(null);
    setEditingTexts({});
    setEditingIndex(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const endpoint = recognitionEngine === "m4" ? "/analyze-m4" : "/analyze";
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setFilename(data.filename);
        setDocId(data.doc_id);
        setResultEngine(data.engine || recognitionEngine);
        const analyzedElements = data.elements.map(item => ({
          ...item,
          engine: item.engine || data.engine || recognitionEngine,
        }));
        setElements(analyzedElements);
        setRecognitionBoxes(analyzedElements);
        setMarkdown(data.markdown);
        setAnnotatedImg(data.annotated_image);
        
        // Populate edit state
        const texts = {};
        analyzedElements.forEach(el => {
          texts[el.index] = el.text;
        });
        setEditingTexts(texts);
      } else {
        const errText = await response.text();
        let errorMessage = "Nhận dạng và phân tích ảnh thất bại.";
        try {
          const errorData = JSON.parse(errText);
          errorMessage = errorData.detail || errorMessage;
        } catch {
          errorMessage = errText || errorMessage;
        }
        setUploadError(errorMessage);
      }
    } catch (error) {
      console.error("Upload / Analyze error:", error);
      setUploadError("Không thể kết nối đến máy chủ nhận dạng.");
    } finally {
      setIsUploading(false);
      setLoading(false);
    }
  };

  const getImagePoint = (event) => {
    const svg = event.currentTarget;
    const rect = svg.getBoundingClientRect();
    const width = imageSize.width || 1;
    const height = imageSize.height || 1;
    return {
      x: Math.max(0, Math.min(width, ((event.clientX - rect.left) / rect.width) * width)),
      y: Math.max(0, Math.min(height, ((event.clientY - rect.top) / rect.height) * height)),
    };
  };

  const handleRegionPointerDown = (event) => {
    if (!isRegionMode) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = getImagePoint(event);
    setCustomBox(null);
    setDrawingBox({ startX: point.x, startY: point.y, endX: point.x, endY: point.y });
  };

  const handleRegionPointerMove = (event) => {
    if (!isRegionMode || !drawingBox) return;
    const point = getImagePoint(event);
    setDrawingBox(prev => ({ ...prev, endX: point.x, endY: point.y }));
  };

  const handleRegionPointerUp = (event) => {
    if (!isRegionMode || !drawingBox) return;
    const point = getImagePoint(event);
    const x1 = Math.round(Math.min(drawingBox.startX, point.x));
    const y1 = Math.round(Math.min(drawingBox.startY, point.y));
    const x2 = Math.round(Math.max(drawingBox.startX, point.x));
    const y2 = Math.round(Math.max(drawingBox.startY, point.y));
    setDrawingBox(null);
    if (x2 - x1 >= 10 && y2 - y1 >= 10) {
      const completedBox = [x1, y1, x2, y2];
      setCustomBox(completedBox);
      scanSelectedRegion(completedBox);
    }
  };

  const cropSelectedRegion = async (box) => {
    if (!box || !originalImg || !imageRef.current) {
      throw new Error("Chưa có vùng ảnh hợp lệ để quét.");
    }

    const source = new Image();
    source.src = originalImg;
    await new Promise((resolve, reject) => {
      source.onload = resolve;
      source.onerror = () => reject(new Error("Không thể đọc ảnh gốc."));
    });

    const [x1, y1, x2, y2] = box;
    const displayWidth = imageSize.width;
    const displayHeight = imageSize.height;
    const scaleX = source.naturalWidth / displayWidth;
    const scaleY = source.naturalHeight / displayHeight;
    const sourceX = Math.round(x1 * scaleX);
    const sourceY = Math.round(y1 * scaleY);
    const cropWidth = Math.max(1, Math.round((x2 - x1) * scaleX));
    const cropHeight = Math.max(1, Math.round((y2 - y1) * scaleY));

    const canvas = document.createElement("canvas");
    canvas.width = cropWidth;
    canvas.height = cropHeight;
    const context = canvas.getContext("2d");
    context.drawImage(
      source,
      sourceX,
      sourceY,
      cropWidth,
      cropHeight,
      0,
      0,
      cropWidth,
      cropHeight,
    );

    return new Promise((resolve, reject) => {
      canvas.toBlob(
        blob => blob ? resolve(blob) : reject(new Error("Không thể tạo ảnh vùng đã chọn.")),
        "image/png",
      );
    });
  };

  const scanSelectedRegion = async (boxOverride = null) => {
    const targetBox = Array.isArray(boxOverride) ? boxOverride : customBox;
    if (!targetBox || regionScanning) return;
    setCustomBox(targetBox);
    setRegionScanning(true);
    setUploadError("");

    try {
      const croppedBlob = await cropSelectedRegion(targetBox);
      const formData = new FormData();
      formData.append("file", croppedBlob, "vung_tuy_chon.png");
      const endpoint = recognitionEngine === "m4" ? "/analyze-m4" : "/analyze";
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Không thể quét vùng đã chọn.");
      }

      const [x1, y1, x2, y2] = targetBox;
      const regionWidth = x2 - x1;
      const regionHeight = y2 - y1;
      const firstIndex = elements.length;
      const regionalElements = data.elements.map((item, offset) => {
        const [localX1, localY1, localX2, localY2] = item.box;
        return {
          ...item,
          index: firstIndex + offset,
          id: `custom-region-${Date.now()}-${offset}`,
          source: "CUSTOM_REGION",
          engine: item.engine || data.engine || recognitionEngine,
          box: [
            Math.round(x1 + (localX1 / data.width) * regionWidth),
            Math.round(y1 + (localY1 / data.height) * regionHeight),
            Math.round(x1 + (localX2 / data.width) * regionWidth),
            Math.round(y1 + (localY2 / data.height) * regionHeight),
          ],
        };
      });

      setElements(prev => {
        const next = reindexElements([...prev, ...regionalElements]);
        setEditingTexts(rebuildEditingTexts(next));
        setMarkdown(buildMarkdownFromElements(next));
        return next;
      });
      setRecognitionBoxes(prev => {
        const knownIds = new Set(prev.map(item => item.id));
        return [...prev, ...regionalElements.filter(item => !knownIds.has(item.id))];
      });
      setSelectedIdx(firstIndex);
      setActiveTab("elements");
      setIsRegionMode(false);
    } catch (error) {
      setUploadError(error.message || "Không thể quét vùng đã chọn.");
    } finally {
      setRegionScanning(false);
    }
  };

  const visibleSelectionBox = drawingBox
    ? [
        Math.min(drawingBox.startX, drawingBox.endX),
        Math.min(drawingBox.startY, drawingBox.endY),
        Math.max(drawingBox.startX, drawingBox.endX),
        Math.max(drawingBox.startY, drawingBox.endY),
      ]
    : customBox;

  const buildMarkdownFromElements = (items) => {
    return items.map(item => {
      if (item.type === "FORMULA") return `$$\n${item.text}\n$$`;
      if (item.type === "TITLE") return `## ${item.text}`;
      return item.text || "";
    }).filter(Boolean).join("\n\n");
  };

  const reindexElements = (items) => {
    return items.map((item, index) => ({ ...item, index }));
  };

  const rebuildEditingTexts = (items) => {
    const next = {};
    items.forEach(item => {
      next[item.index] = item.text;
    });
    return next;
  };

  const deleteResult = (index) => {
    setElements(prev => {
      const next = reindexElements(prev.filter(item => item.index !== index));
      setEditingTexts(rebuildEditingTexts(next));
      setMarkdown(buildMarkdownFromElements(next));
      return next;
    });
    setSelectedIdx(null);
    setEditingIndex(null);
  };

  const clearAllResults = () => {
    setElements([]);
    setEditingTexts({});
    setMarkdown("");
    setSelectedIdx(null);
    setEditingIndex(null);
    setCustomBox(null);
    setDrawingBox(null);
  };

  const copyElementText = (index, text) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const saveElementText = (index) => {
    setElements(prev => {
      const next = prev.map(el => {
        if (el.index === index) {
          return { ...el, text: editingTexts[index] };
        }
        return el;
      });
      setMarkdown(buildMarkdownFromElements(next));
      return next;
    });
    setEditingIndex(null);
  };

  const copyMarkdownText = () => {
    navigator.clipboard.writeText(markdown);
    setMarkdownCopied(true);
    setTimeout(() => setMarkdownCopied(false), 2000);
  };

  const downloadMarkdownFile = () => {
    const element = document.createElement("a");
    const file = new Blob([markdown], { type: "text/markdown" });
    element.href = URL.createObjectURL(file);
    const engineSuffix = resultEngine === "m4" ? "m4" : "pix2text";
    element.download = `${filename.split(".")[0] || "tai_lieu"}_${engineSuffix}.md`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Maps Element types to customized colored classes
  const getBadgeClass = (typeName) => {
    if (typeName === "FORMULA") return "badge-formula";
    if (typeName === "TEXT") return "badge-text";
    if (typeName === "TITLE") return "badge-title";
    if (typeName === "TABLE") return "badge-table";
    return "badge-default";
  };

  return (
    <div className="d-flex flex-column min-vh-100 pb-5" style={{ backgroundColor: '#f8fafc' }}>
      
      {/* 1. Header with Light Accents and Large Text */}
      <header className="navbar border-bottom bg-white px-4 py-3 mb-4 shadow-sm rounded-0 glass-panel">
        <div className="container-fluid d-flex justify-content-between align-items-center">
          <div className="d-flex align-items-center gap-3">
            <div className="p-2 bg-gradient-brand rounded-3 shadow-lg d-flex align-items-center justify-content-center">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
            <div>
              <div className="d-flex align-items-center gap-2">
                <h1 className="h3 fw-bold m-0 bg-gradient-text" style={{ fontSize: '1.8rem' }}>
                  Pix2Text Native Studio
                </h1>
                <span className="badge bg-primary bg-opacity-10 text-primary border border-primary border-opacity-20 fw-bold px-2 py-1" style={{ fontSize: '10px' }}>
                  V1.1 Engine
                </span>
              </div>
              <p className="text-secondary m-0 mt-0.5" style={{ fontSize: '13px' }}>
                Hệ thống OCR tài liệu và công thức toán học Breezedeus Pix2Text nguyên bản (GPU Accelerated)
              </p>
            </div>
          </div>

          {/* GPU online indicator */}
          <div>
            {systemHealth ? (
              systemHealth.status === "healthy" ? (
                <div className="d-flex align-items-center gap-2">
                  <span className="badge bg-success bg-opacity-10 text-success border border-success border-opacity-20 rounded-pill py-2 px-3.5 fs-6 fw-bold">
                    <CheckCircle className="w-4 h-4 me-1 d-inline-block align-middle" />
                    GPU: {systemHealth.device.toUpperCase()} Active
                  </span>
                </div>
              ) : (
                <div className="d-flex align-items-center gap-2 bg-danger bg-opacity-10 border border-danger border-opacity-20 rounded-pill py-2 px-3.5 text-danger fs-6 fw-bold">
                  <span className="dot bg-danger"></span>
                  <span>Pix2Text Offline</span>
                </div>
              )
            ) : (
              <span className="text-secondary fs-6">Đang kết nối backend...</span>
            )}
          </div>
        </div>
      </header>

      {/* 2. Workboard Workspace */}
      <main className="container-xxl px-4 mt-2">
        <div className="row g-4">
          
          {/* LEFT COLUMN: Upload, Document Canvas & Stats (col-lg-5) */}
          <div className="col-12 col-lg-5 d-flex flex-column gap-4">
            
            {/* Uploader Card */}
            <div className="glass-panel p-4">
              <h2 className="h4 fw-bold text-dark mb-3 d-flex align-items-center gap-2">
                <Upload className="w-6 h-6 text-primary" />
                Tải lên hình ảnh
              </h2>

              <div className="mb-4">
                <p className="uppercase-title mb-2" style={{ fontSize: '10px' }}>
                  Chọn mô hình nhận dạng
                </p>
                <div className="engine-selector">
                  <button
                    type="button"
                    className={`engine-option ${recognitionEngine === "pix2text" ? "active" : ""}`}
                    onClick={() => setRecognitionEngine("pix2text")}
                  >
                    <LayoutGrid className="w-5 h-5" />
                    <span>
                      <strong>Pix2Text</strong>
                      <small>Phân tích toàn bộ tài liệu, văn bản, bảng và công thức</small>
                    </span>
                  </button>
                  <button
                    type="button"
                    className={`engine-option ${recognitionEngine === "m4" ? "active" : ""}`}
                    onClick={() => setRecognitionEngine("m4")}
                    disabled={systemHealth?.m4_available === false}
                    title={systemHealth?.m4_available === false ? "Không tìm thấy checkpoint M4" : ""}
                  >
                    <Network className="w-5 h-5" />
                    <span>
                      <strong>M4 Coordinate-Aware GAT</strong>
                      <small>Nhận dạng một công thức toán học viết tay thành LaTeX</small>
                    </span>
                  </button>
                </div>
                {recognitionEngine === "m4" && (
                  <div className="alert alert-primary bg-primary bg-opacity-10 border-0 mt-2 mb-0 py-2 px-3 small">
                    M4 phù hợp với ảnh chỉ chứa một biểu thức viết tay. Hãy cắt riêng công thức để có kết quả tốt nhất.
                  </div>
                )}
              </div>
              
              <label className="d-flex flex-column align-items-center justify-content-center border-dashed rounded-3 p-4 text-center cursor-pointer hover-brand-border mb-3" style={{ border: '2px dashed rgba(0,0,0,0.12)', minHeight: '160px', backgroundColor: '#f8fafc' }}>
                <input type="file" accept="image/png, image/jpeg, image/jpg, image/webp" className="d-none" onChange={handleFileSelection} />
                <div className="p-3 bg-white rounded-circle border shadow-sm mb-3">
                  <Upload className="w-8 h-8 text-primary" />
                </div>
                <div>
                  <p className="fs-5 fw-bold m-0 text-dark">
                    Nhấp hoặc kéo thả ảnh vào đây
                  </p>
                  <p className="text-secondary m-0 mt-1" style={{ fontSize: '12px' }}>Hỗ trợ các tệp ảnh PNG, JPG, JPEG, WEBP</p>
                </div>
              </label>

              <button
                type="button"
                className="btn btn-gradient w-100 d-flex align-items-center justify-content-center gap-2 mb-3"
                onClick={handleImageProcessing}
                disabled={!selectedFile || isUploading}
              >
                {isUploading
                  ? <RefreshCw className="w-5 h-5 animate-spin" />
                  : recognitionEngine === "m4"
                    ? <Network className="w-5 h-5" />
                    : <ScanLine className="w-5 h-5" />}
                {isUploading
                  ? "Đang xử lý ảnh..."
                  : `Xử lý ảnh bằng ${recognitionEngine === "m4" ? "M4" : "Pix2Text"}`}
              </button>

              {isUploading && (
                <div className="d-flex align-items-center gap-2 justify-content-center text-primary fs-6 py-2 mb-2 fw-bold">
                  <RefreshCw className="w-4 h-4 spinner-border spinner-border-sm border-0 animate-spin" />
                  <span>
                    {recognitionEngine === "m4"
                      ? "Đang nhận dạng công thức bằng mô hình M4..."
                      : "Đang tải lên và phân tích bố cục..."}
                  </span>
                </div>
              )}
              
              {uploadError && (
                <div className="alert alert-danger p-3 fs-6 border-0 mb-3 bg-danger bg-opacity-10 text-danger rounded-3 d-flex align-items-center gap-2 fw-semibold">
                  <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                  <span>{uploadError}</span>
                </div>
              )}

              {selectedFile && (
                <div className="d-flex align-items-center gap-3 p-3 bg-light rounded-3 border border-secondary border-opacity-10">
                  <FileText className="w-10 h-10 text-primary flex-shrink-0" />
                  <div className="overflow-hidden">
                    <p className="m-0 text-muted uppercase-title" style={{ fontSize: '9px' }}>Tệp đã chọn</p>
                    <p className="m-0 text-dark fs-6 fw-bold text-truncate" title={filename}>{filename}</p>
                    <p className="m-0 text-secondary" style={{ fontSize: '11px' }}>
                      Chuyển mô hình nếu cần, sau đó bấm nút xử lý.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Document Canvas and SVG box outlines */}
            {docId && (
              <div className="glass-panel p-4 d-flex flex-column gap-3">
                <h3 className="h5 fw-bold text-dark m-0 d-flex justify-content-between align-items-center">
                  <span>Khung Nhìn Ảnh Phân Tích</span>
                  <span className="badge bg-primary text-white fs-6 py-1.5 px-3 rounded-pill fw-bold">
                    {resultEngine === "m4" ? "Mô hình M4" : `${elements.length} phân đoạn`}
                  </span>
                </h3>

                <div className="region-toolbar">
                  <button
                    type="button"
                    className={`btn btn-sm d-flex align-items-center gap-2 ${isRegionMode ? "btn-primary" : "btn-outline-primary"}`}
                    onClick={() => {
                      setIsRegionMode(prev => !prev);
                      setDrawingBox(null);
                    }}
                  >
                    <ScanLine className="w-4 h-4" />
                    {isRegionMode ? "Đang chọn vùng" : "Chọn vùng tùy ý"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-success d-flex align-items-center gap-2"
                    disabled={!customBox || regionScanning}
                    onClick={() => scanSelectedRegion()}
                  >
                    {regionScanning
                      ? <RefreshCw className="w-4 h-4 animate-spin" />
                      : <ScanLine className="w-4 h-4" />}
                    Quét vùng đã chọn bằng {recognitionEngine === "m4" ? "M4" : "Pix2Text"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-danger d-flex align-items-center gap-2"
                    disabled={!customBox && !drawingBox}
                    onClick={() => {
                      setCustomBox(null);
                      setDrawingBox(null);
                    }}
                  >
                    <Trash2 className="w-4 h-4" />
                    Xóa vùng
                  </button>
                </div>

                {isRegionMode && (
                  <div className="small text-primary bg-primary bg-opacity-10 rounded-3 px-3 py-2">
                    Giữ chuột và kéo trực tiếp trên ảnh. Hệ thống sẽ tự nhận dạng ngay khi bạn thả chuột.
                  </div>
                )}
                {!isRegionMode && (
                  <div className="small text-secondary bg-light rounded-3 px-3 py-2 border">
                    Nhấp trực tiếp vào một bounding box để nhận dạng lại vùng đó bằng mô hình đang chọn.
                  </div>
                )}

                <div 
                  className="position-relative overflow-auto border rounded-3 bg-light p-2 d-flex align-items-start justify-content-center"
                  style={{ maxHeight: '580px' }}
                >
                  {(loading || regionScanning) && (
                    <div className="position-absolute top-0 start-0 w-100 h-100 bg-white bg-opacity-75 z-3 d-flex flex-column align-items-center justify-content-center gap-2">
                      <RefreshCw className="w-10 h-10 text-primary animate-spin" />
                      <p className="fs-5 text-dark fw-bold m-0">
                        {regionScanning
                          ? `Đang quét vùng tùy chọn bằng ${recognitionEngine === "m4" ? "M4" : "Pix2Text"}...`
                          : recognitionEngine === "m4"
                          ? "Đang chạy M4 Coordinate-Aware GAT trên GPU..."
                          : "Đang chạy MFD Layout YOLO trên GPU..."}
                      </p>
                    </div>
                  )}

                  {annotatedImg ? (
                    <div className="position-relative">
                      <img 
                        ref={imageRef}
                        src={annotatedImg} 
                        alt="Bản đồ phân tích layout" 
                        className="img-fluid rounded-3 shadow-sm"
                        style={{ userSelect: 'none' }}
                        draggable="false"
                        onLoad={(event) => {
                          setImageSize({
                            width: event.currentTarget.naturalWidth,
                            height: event.currentTarget.naturalHeight,
                          });
                        }}
                      />

                      {/* SVG Invisible click triggers overlay */}
                      {imageSize.width > 0 && imageSize.height > 0 && (
                        <svg 
                          className={`position-absolute top-0 start-0 w-100 h-100 ${isRegionMode ? "region-drawing-layer" : "pointer-events-none"}`}
                          viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                          onPointerDown={handleRegionPointerDown}
                          onPointerMove={handleRegionPointerMove}
                          onPointerUp={handleRegionPointerUp}
                          onPointerCancel={() => setDrawingBox(null)}
                        >
                          {recognitionBoxes.map((el) => {
                            const [xmin, ymin, xmax, ymax] = el.box;
                            const isSelected = elements.some(
                              item => item.index === selectedIdx && item.id === el.id
                            );
                            return (
                              <g
                                key={`bbox-trigger-${el.id || el.index}`}
                                className={isRegionMode ? "" : "pointer-events-auto cursor-pointer"}
                                onClick={() => {
                                if (isRegionMode) return;
                                setActiveTab("elements");
                                scanSelectedRegion(el.box);
                              }}
                              >
                                <rect
                                  x={xmin}
                                  y={ymin}
                                  width={xmax - xmin}
                                  height={ymax - ymin}
                                  style={{
                                    fill: isSelected ? 'rgba(37, 99, 235, 0.08)' : 'transparent',
                                    stroke: isSelected ? '#2563eb' : 'transparent',
                                    strokeWidth: isSelected ? 4 : 0,
                                    transition: 'all 0.1s ease'
                                  }}
                                />
                              </g>
                            );
                          })}
                          {visibleSelectionBox && (
                            <g className="custom-region-box">
                              <rect
                                x={visibleSelectionBox[0]}
                                y={visibleSelectionBox[1]}
                                width={visibleSelectionBox[2] - visibleSelectionBox[0]}
                                height={visibleSelectionBox[3] - visibleSelectionBox[1]}
                              />
                              <text
                                x={visibleSelectionBox[0] + 8}
                                y={Math.max(20, visibleSelectionBox[1] - 8)}
                              >
                                Vùng tùy chọn
                              </text>
                            </g>
                          )}
                        </svg>
                      )}
                    </div>
                  ) : null}
                </div>
              </div>
            )}
          </div>

          {/* RIGHT COLUMN: Tab switcher with detailed output cards & Markdown parser (col-lg-7) */}
          <div className="col-12 col-lg-7">
            {docId ? (
              <div className="glass-panel p-4 d-flex flex-column gap-4 col-scroll">
                
                {/* Main Action Tab switchers */}
                <ul className="nav custom-nav-tabs justify-content-start w-100">
                  <li className="nav-item">
                    <button 
                      className={`nav-link d-flex align-items-center gap-2 ${activeTab === "elements" ? "active" : ""}`}
                      onClick={() => setActiveTab("elements")}
                    >
                      <ListCollapse className="w-5 h-5" />
                      <span>Chi tiết các Phân Đoạn ({elements.length})</span>
                    </button>
                  </li>
                  <li className="nav-item">
                    <button 
                      className={`nav-link d-flex align-items-center gap-2 ${activeTab === "markdown" ? "active" : ""}`}
                      onClick={() => setActiveTab("markdown")}
                    >
                      <BookOpen className="w-5 h-5" />
                      <span>Bản Dịch Toàn Văn (Markdown)</span>
                    </button>
                  </li>
                </ul>

                {/* TAB 1 CONTENT: Detailed Grid of all elements */}
                {activeTab === "elements" && (
                  <div className="d-flex flex-column gap-3.5">
                    <div className="d-flex flex-wrap justify-content-between align-items-center gap-2 bg-light border rounded-3 px-3 py-2">
                      <span className="text-secondary small">
                        Đang quản lý {elements.length} kết quả nhận dạng
                      </span>
                      <button
                        type="button"
                        className="btn btn-sm btn-outline-danger d-flex align-items-center gap-2"
                        onClick={clearAllResults}
                        disabled={elements.length === 0}
                      >
                        <Trash2 className="w-4 h-4" />
                        Xóa toàn bộ kết quả
                      </button>
                    </div>

                    {elements.length === 0 && (
                      <div className="text-center text-secondary border border-dashed rounded-3 p-5">
                        Chưa có kết quả. Hãy click vào bounding box hoặc kéo một vùng mới để nhận dạng lại.
                      </div>
                    )}

                    {elements.map((el) => {
                      const isSelected = selectedIdx === el.index;
                      const isEditing = editingIndex === el.index;
                      
                      return (
                        <div 
                          key={el.id || `result-${el.index}`}
                          id={`element-card-${el.index}`}
                          className={`element-card fade-in ${isSelected ? "selected" : ""}`}
                          onClick={() => setSelectedIdx(el.index)}
                        >
                          {/* Card header */}
                          <div className="d-flex flex-wrap justify-content-between align-items-center border-bottom pb-3 mb-3 gap-2">
                            <div className="d-flex align-items-center gap-2.5">
                              <span className="fs-5 fw-bold text-dark font-monospace">#{el.index + 1}</span>
                              <span className={`badge ${getBadgeClass(el.type)} px-2.5 py-1 text-uppercase`}>
                                {el.type}
                              </span>
                              {el.source === "CUSTOM_REGION" && (
                                <span className="badge bg-primary bg-opacity-10 text-primary border border-primary border-opacity-25">
                                  Vùng tùy chọn
                                </span>
                              )}
                              <span className="text-secondary font-monospace" style={{ fontSize: '12px' }}>
                                {el.engine === "m4" ? "Điểm mô hình" : "Độ tin cậy"}: {el.score.toFixed(2)}
                              </span>
                            </div>
                            
                            {/* Copy & Edit tools */}
                            <div className="d-flex flex-wrap gap-3">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  if (isEditing) {
                                    saveElementText(el.index);
                                  } else {
                                    setEditingIndex(el.index);
                                  }
                                }}
                                className="btn btn-link text-decoration-none text-primary p-0 d-flex align-items-center gap-1.5 fw-bold fs-6"
                              >
                                <Edit className="w-4 h-4" />
                                {isEditing ? "Lưu lại" : "Sửa"}
                              </button>
                              
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  copyElementText(el.index, el.text);
                                }}
                                className="btn btn-link text-decoration-none text-secondary p-0 d-flex align-items-center gap-1.5 fw-bold fs-6"
                              >
                                {copiedIndex === el.index ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4 text-secondary" />}
                                <span className={copiedIndex === el.index ? "text-success" : "text-secondary"}>
                                  {copiedIndex === el.index ? "Đã copy" : "Copy"}
                                </span>
                              </button>

                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  deleteResult(el.index);
                                }}
                                className="btn btn-link text-decoration-none text-danger p-0 d-flex align-items-center gap-1.5 fw-bold fs-6"
                                title={`Xóa kết quả #${el.index + 1}`}
                              >
                                <Trash2 className="w-4 h-4" />
                                Xóa
                              </button>
                            </div>
                          </div>

                          {/* Body: Split View Crop Preview vs Recognized output */}
                          <div className="row g-3 align-items-start">
                            
                            {/* Left part inside card: Crop preview */}
                            <div className="col-12 col-md-3 text-center text-md-start">
                              <p className="uppercase-title mb-2" style={{ fontSize: '9px' }}>Ảnh phân tách</p>
                              <div className="crop-preview-frame">
                                <img 
                                  src={el.image} 
                                  alt={`Phân đoạn #${el.index}`} 
                                  className="img-fluid rounded" 
                                  style={{ maxHeight: '100px', objectFit: 'contain' }} 
                                />
                              </div>
                              <p className="text-secondary font-monospace mt-1.5" style={{ fontSize: '10px' }}>
                                {el.box[2] - el.box[0]}x{el.box[3] - el.box[1]}px
                              </p>
                            </div>

                            {/* Right part inside card: Parsed text / math visual output */}
                            <div className="col-12 col-md-9 border-start border-light ps-md-4">
                              <p className="uppercase-title mb-2" style={{ fontSize: '9px' }}>Kết quả nhận diện</p>
                              
                              {isEditing ? (
                                <textarea
                                  value={editingTexts[el.index] || ""}
                                  onChange={(e) => setEditingTexts({ ...editingTexts, [el.index]: e.target.value })}
                                  rows={3}
                                  className="form-control border bg-light text-dark font-monospace fs-5 py-2 px-3"
                                />
                              ) : (
                                <div className="w-100">
                                  {el.type === "FORMULA" ? (
                                    <div className="w-100">
                                      <LaTeXRenderer latex={el.text} />
                                      <pre className="bg-light p-2.5 rounded-3 border border-secondary border-opacity-10 font-monospace fs-6 text-dark mt-2 select-all overflow-x-auto">
                                        {el.text}
                                      </pre>
                                    </div>
                                  ) : (
                                    <div className="bg-light p-3 rounded-3 border border-secondary border-opacity-10 text-dark fs-5 font-monospace whitespace-pre-wrap select-all">
                                      {el.text}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>

                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* TAB 2 CONTENT: Page Markdown report */}
                {activeTab === "markdown" && (
                  <div className="d-flex flex-column gap-3.5">
                    
                    {/* Header bar controls */}
                    <div className="d-flex flex-wrap justify-content-between align-items-center bg-light p-2.5 rounded-3 border gap-3">
                      <div className="btn-group" role="group">
                        <button
                          onClick={() => setRightPanelTab("visual")}
                          className={`btn btn-sm btn-outline-blue uppercase-title py-1.5 px-4 ${rightPanelTab === "visual" ? "active bg-primary text-white fw-bold" : ""}`}
                          style={{ fontSize: '11px' }}
                        >
                          Bản Render Toán Học
                        </button>
                        <button
                          onClick={() => setRightPanelTab("source")}
                          className={`btn btn-sm btn-outline-blue uppercase-title py-1.5 px-4 ${rightPanelTab === "source" ? "active bg-primary text-white fw-bold" : ""}`}
                          style={{ fontSize: '11px' }}
                        >
                          Mã Nguồn Markdown
                        </button>
                      </div>

                      {markdown && (
                        <div className="d-flex gap-2">
                          <button
                            onClick={copyMarkdownText}
                            className="btn btn-sm btn-outline-blue d-flex align-items-center gap-1.5 py-1.5 px-3"
                          >
                            {markdownCopied ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
                            <span className="small">{markdownCopied ? "Đã copy" : "Copy Markdown"}</span>
                          </button>
                          <button
                            onClick={downloadMarkdownFile}
                            className="btn btn-sm btn-outline-blue d-flex align-items-center gap-1.5 py-1.5 px-3"
                          >
                            <Download className="w-4 h-4" />
                            <span className="small">Tải về (.md)</span>
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Rendering contents panel */}
                    <div className="p-3 bg-light bg-opacity-50 rounded-3 border border-secondary border-opacity-10" style={{ minHeight: '400px' }}>
                      {rightPanelTab === "visual" ? (
                        <MarkdownMathRenderer content={markdown} />
                      ) : (
                        <textarea
                          value={markdown}
                          onChange={(e) => setMarkdown(e.target.value)}
                          className="w-100 h-100 bg-transparent border-0 text-dark font-monospace fs-5"
                          style={{ minHeight: '380px', outline: 'none', resize: 'none' }}
                          placeholder="Mã nguồn Markdown sẽ hiển thị ở đây..."
                        />
                      )}
                    </div>

                  </div>
                )}

              </div>
            ) : (
              <div className="glass-panel bg-dark-panel p-5 d-flex flex-column align-items-center justify-content-center text-center" style={{ minHeight: '580px' }}>
                <div className="p-4 bg-white rounded-circle border shadow-sm mb-4 pulse-glow">
                  <BookOpen className="w-16 h-14 text-muted" />
                </div>
                <h2 className="h4 text-dark fw-bold">Trình phân tích tài liệu</h2>
                <p className="fs-5 text-secondary mt-2" style={{ maxWidth: '440px' }}>
                  Chọn Pix2Text để phân tích tài liệu hoặc chọn M4 để nhận dạng một công thức toán học viết tay.
                </p>
              </div>
            )}
          </div>

        </div>
      </main>
      
    </div>
  );
}
