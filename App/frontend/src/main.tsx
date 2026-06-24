import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import { Image as KonvaImage, Layer, Rect, Stage, Text, Transformer } from "react-konva";
import type Konva from "konva";
import {
  BoxSelect,
  Check,
  ChevronsUpDown,
  Download,
  Eye,
  FileJson,
  Hand,
  ImagePlus,
  Layers,
  Maximize,
  MousePointer2,
  ScanLine,
  Scissors,
  Settings,
  Trash2,
  X
} from "lucide-react";
import katex from "katex";
import {
  assetUrl,
  clearPageExpressions,
  createExpression,
  deleteExpression,
  deletePage,
  ensureProject,
  exportCrohmeM4,
  exportProject,
  getPage,
  mergeExpressions,
  normalizeExpression,
  normalizePage,
  patchExpression,
  recognizeAccepted,
  recognizeExpression,
  reorderExpressions,
  scanPage,
  splitExpression,
  updateLatex,
  uploadPages
} from "./api";
import { clampBBox, normalizeBBox } from "./geometry";
import type { BBox, ExpressionBox, ExpressionStatus, LayerKey, PageItem, Project, Tool } from "./types";
import "bootstrap/dist/css/bootstrap.min.css";
import "katex/dist/katex.min.css";
import "./styles.css";

const statusLabel: Record<ExpressionStatus, string> = {
  auto_detected: "Tự phát hiện",
  need_review: "Cần xem lại",
  accepted: "Đã duyệt",
  rejected: "Đã loại",
  edited: "Đã sửa",
  exported: "Đã xuất",
  noise: "Nhiễu",
  fragment: "Mảnh nhỏ"
};

const statusClass: Record<ExpressionStatus, string> = {
  auto_detected: "badge-blue",
  need_review: "badge-yellow",
  accepted: "badge-green",
  rejected: "badge-gray",
  edited: "badge-purple",
  exported: "badge-green",
  noise: "badge-red",
  fragment: "badge-gray"
};

const boxColor: Record<ExpressionStatus, string> = {
  auto_detected: "#2563EB",
  need_review: "#F59E0B",
  accepted: "#16A34A",
  rejected: "#94A3B8",
  edited: "#9333EA",
  exported: "#16A34A",
  noise: "#DC2626",
  fragment: "#64748B"
};

const candidateLabel = {
  single_expression: "Single Expression",
  multiline_block: "Multi-line: cần split",
  fragment: "Fragment",
  noise: "Noise",
  uncertain: "Uncertain"
} as const;

function useImage(url?: string) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!url) {
      setImage(null);
      return;
    }
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.onload = () => setImage(img);
    img.src = assetUrl(url);
  }, [url]);
  return image;
}

function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [currentPageId, setCurrentPageId] = useState<string>();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [tool, setTool] = useState<Tool>("select");
  const [layer, setLayer] = useState<LayerKey>("original");
  const [isBusy, setBusy] = useState(false);
  const [message, setMessage] = useState("Sẵn sàng xử lý ảnh trang giấy.");

  useEffect(() => {
    ensureProject().then((p) => {
      setProject(p);
      setCurrentPageId(p.pages[0]?.id);
    });
  }, []);

  const currentPage = useMemo(() => project?.pages.find((p) => p.id === currentPageId), [project, currentPageId]);
  const selectedExpressions = useMemo(
    () => currentPage?.expressions.filter((expr) => selectedIds.includes(expr.id)) ?? [],
    [currentPage, selectedIds]
  );
  const selectedExpression = useMemo(() => {
    const activeId = selectedIds[selectedIds.length - 1];
    return currentPage?.expressions.find((expr) => expr.id === activeId);
  }, [currentPage, selectedIds]);

  const refreshPage = async (pageId = currentPageId) => {
    if (!project || !pageId) return;
    const page = await getPage(pageId);
    setProject((current) => {
      if (!current) return current;
      return { ...current, pages: current.pages.map((item) => (item.id === page.id ? page : item)) };
    });
  };

  const handleUpload = async (files: FileList | null) => {
    if (!project || !files?.length) return;
    setBusy(true);
    setMessage("Đang tải ảnh lên backend...");
    const pages = await uploadPages(project.id, files);
    const nextProject = { ...project, pages: [...project.pages, ...pages] };
    setProject(nextProject);
    setCurrentPageId(pages[0].id);
    setSelectedIds([]);
    setMessage(`Đã tải ${pages.length} ảnh. Chọn Auto Scan để phát hiện biểu thức.`);
    setBusy(false);
  };

  const handleScan = async () => {
    if (!currentPage) return;
    setBusy(true);
    setMessage("Đang quét ảnh bằng OpenCV và gom connected components...");
    await scanPage(currentPage.id);
    await refreshPage(currentPage.id);
    setSelectedIds([]);
    setMessage("Đã quét xong. Kiểm tra các bbox màu vàng trước khi xuất.");
    setBusy(false);
  };

  const updateExpr = async (id: string, payload: { bbox?: BBox; status?: ExpressionStatus }) => {
    await patchExpression(id, payload);
    await refreshPage();
  };

  const setStatus = async (status: ExpressionStatus) => {
    setBusy(true);
    for (const expr of selectedExpressions) await patchExpression(expr.id, { status });
    await refreshPage();
    setMessage(status === "accepted" ? "Đã duyệt biểu thức đã chọn." : "Đã cập nhật trạng thái biểu thức.");
    setBusy(false);
  };

  const rejectOrDelete = async () => {
    setBusy(true);
    for (const id of selectedIds) await patchExpression(id, { status: "rejected" });
    await refreshPage();
    setMessage("Đã đánh dấu loại bỏ bbox đã chọn.");
    setBusy(false);
  };

  const hardDelete = async () => {
    setBusy(true);
    for (const id of selectedIds) await deleteExpression(id);
    setSelectedIds([]);
    await refreshPage();
    setMessage("Đã xóa bbox khỏi trang.");
    setBusy(false);
  };

  const handleClearQueue = async () => {
    if (!currentPage) return;
    if (!window.confirm(`Xóa toàn bộ Expression Queue của "${currentPage.fileName}"? Thao tác này sẽ xóa cả crop/normalized/debug của page này.`)) return;
    setBusy(true);
    await clearPageExpressions(currentPage.id);
    await refreshPage(currentPage.id);
    setSelectedIds([]);
    setMessage("Đã xóa toàn bộ Expression Queue của page hiện tại.");
    setBusy(false);
  };

  const handleDeletePage = async (pageId: string) => {
    if (!project) return;
    const page = project.pages.find((item) => item.id === pageId);
    if (!page) return;
    if (!window.confirm(`Xóa ảnh đã upload "${page.fileName}" và toàn bộ dữ liệu liên quan?`)) return;
    setBusy(true);
    await deletePage(pageId);
    const nextPages = project.pages.filter((item) => item.id !== pageId);
    setProject({ ...project, pages: nextPages });
    setCurrentPageId((current) => (current === pageId ? nextPages[0]?.id : current));
    setSelectedIds([]);
    setMessage("Đã xóa ảnh upload và toàn bộ dữ liệu liên quan.");
    setBusy(false);
  };

  const handleMerge = async () => {
    if (!currentPage || selectedIds.length < 2) return;
    setBusy(true);
    const result = await mergeExpressions(currentPage.id, selectedIds);
    setSelectedIds([result.mergedExpression.id]);
    await refreshPage();
    setMessage("Đã gộp các bbox được chọn thành một biểu thức.");
    setBusy(false);
  };

  const handleSplit = async (mode: "horizontal" | "vertical") => {
    if (!selectedExpression) return;
    setBusy(true);
    const result = await splitExpression(selectedExpression.id, mode, 0.5);
    setSelectedIds(result.createdExpressions.map((expr) => expr.id));
    await refreshPage();
    setMessage(mode === "horizontal" ? "Đã tách bbox theo chiều ngang." : "Đã tách bbox theo chiều dọc.");
    setBusy(false);
  };

  const handleExport = async () => {
    if (!project) return;
    setBusy(true);
    const url = await exportProject(project.id);
    window.open(assetUrl(url), "_blank");
    setMessage("Đã xuất metadata và crop cho các biểu thức đã duyệt.");
    setBusy(false);
  };

  const handleNormalizeAll = async () => {
    if (!currentPage) return;
    setBusy(true);
    setMessage("Đang chuẩn hóa các expression thành ảnh M4-ready nền đen chữ trắng...");
    const normalized = await normalizePage(currentPage.id);
    await refreshPage(currentPage.id);
    setMessage(`Đã normalize ${normalized.length} expression sang chuẩn CROHME-like/M4-ready.`);
    setBusy(false);
  };

  const handleRecognizeAccepted = async () => {
    if (!currentPage) return;
    setBusy(true);
    setMessage("Đang chạy M4 cho các expression đã accepted...");
    const recognized = await recognizeAccepted(currentPage.id);
    await refreshPage(currentPage.id);
    setMessage(`Đã chạy M4 cho ${recognized.length} expression accepted.`);
    setBusy(false);
  };

  const handleExportM4 = async () => {
    if (!project) return;
    setBusy(true);
    const url = await exportCrohmeM4(project.id, "accepted_recognized");
    window.open(assetUrl(url), "_blank");
    setMessage("Đã xuất dataset CROHME-like/M4-ready.");
    setBusy(false);
  };

  const stats = useMemo(() => {
    const pages = project?.pages ?? [];
    const expressions = pages.flatMap((p) => p.expressions);
    return {
      pages: pages.length,
      expressions: expressions.length,
      accepted: expressions.filter((e) => e.status === "accepted").length,
      review: expressions.filter((e) => e.status === "need_review").length,
      rejected: expressions.filter((e) => e.status === "rejected").length
    };
  }, [project]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return;
      if (event.key === "v" || event.key === "V") setTool("select");
      if (event.key === "b" || event.key === "B") setTool("draw");
      if (event.key === " ") setTool("pan");
      if (event.key === "a" || event.key === "A") void setStatus("accepted");
      if (event.key === "r" || event.key === "R") void setStatus("rejected");
      if (event.key === "Delete") void rejectOrDelete();
      if (event.key === "m" || event.key === "M") void handleMerge();
      if (event.key === "s" || event.key === "S") void handleSplit("horizontal");
      if (event.key >= "1" && event.key <= "4") setLayer(["original", "cleaned", "binary", "components"][Number(event.key) - 1] as LayerKey);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!project) return <div className="loading">Đang mở Expression Page Explorer...</div>;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Expression Page Explorer</h1>
          <p>Project: {project.name} · Page Splitter Workspace</p>
        </div>
        <div className="top-actions">
          <label className="button primary">
            <ImagePlus size={17} />
            Upload Images
            <input hidden type="file" accept="image/*" multiple onChange={(e) => void handleUpload(e.target.files)} />
          </label>
          <button className="button" onClick={handleScan} disabled={!currentPage || isBusy}>
            <ScanLine size={17} /> Auto Scan
          </button>
          <button className="button" onClick={() => setSelectedIds(currentPage?.expressions.filter((e) => e.status === "need_review").map((e) => e.id) ?? [])}>
            <Eye size={17} /> Review Warnings
          </button>
          <button className="button" onClick={handleExport}>
            <Download size={17} /> Export
          </button>
          <button className="button" onClick={handleNormalizeAll} disabled={!currentPage || isBusy}>
            <FileJson size={17} /> Normalize All
          </button>
          <button className="button" onClick={handleRecognizeAccepted} disabled={!currentPage || isBusy}>
            <ScanLine size={17} /> Run M4 Accepted
          </button>
          <button className="button" onClick={handleExportM4} disabled={isBusy}>
            <Download size={17} /> Export M4 Dataset
          </button>
          <button className="button ghost">
            <Settings size={17} /> Settings
          </button>
        </div>
        <div className="stats">
          Pages: {stats.pages} | Expressions: {stats.expressions} | Accepted: {stats.accepted} | Need Review: {stats.review} | Rejected: {stats.rejected}
        </div>
      </header>

      <main className="workspace">
        <aside className="sidebar">
          <div className="panel-title">
            <span>Page List</span>
            <small>All pages</small>
          </div>
          {project.pages.length === 0 ? (
            <div className="empty">Chưa có ảnh. Hãy upload ảnh trang giấy A4 để bắt đầu.</div>
          ) : (
            project.pages.map((page) => (
              <button key={page.id} className={`page-card ${page.id === currentPageId ? "active" : ""}`} onClick={() => { setCurrentPageId(page.id); setSelectedIds([]); }}>
                <div>
                  <strong>{page.fileName}</strong>
                  <span>{page.expressions.length} expressions</span>
                  <small>{page.expressions.filter((e) => e.status === "accepted").length} accepted | {page.expressions.filter((e) => e.status === "need_review").length} warn</small>
                </div>
                {page.thumbnailUrl && <img src={assetUrl(page.thumbnailUrl)} alt={page.fileName} />}
                <em className={`page-status ${page.status}`}>{page.status}</em>
                <span
                  className="page-delete"
                  role="button"
                  tabIndex={0}
                  title="Xóa ảnh upload"
                  onClick={(event) => {
                    event.stopPropagation();
                    void handleDeletePage(page.id);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.stopPropagation();
                      void handleDeletePage(page.id);
                    }
                  }}
                >
                  <Trash2 size={14} /> Xóa ảnh
                </span>
              </button>
            ))
          )}
        </aside>

        <section className="viewer-section">
          <Toolbar
            tool={tool}
            layer={layer}
            selectedCount={selectedIds.length}
            onTool={setTool}
            onLayer={setLayer}
            onAccept={() => void setStatus("accepted")}
            onReject={() => void rejectOrDelete()}
            onDelete={() => void hardDelete()}
            onClearQueue={() => void handleClearQueue()}
            onMerge={() => void handleMerge()}
            onSplit={handleSplit}
          />
          <PageViewer
            page={currentPage}
            selectedIds={selectedIds}
            tool={tool}
            layer={layer}
            onSelect={setSelectedIds}
            onCreate={async (bbox) => {
              if (!currentPage) return;
              const expr = await createExpression(currentPage.id, bbox);
              setSelectedIds([expr.id]);
              await refreshPage(currentPage.id);
            }}
            onPatch={updateExpr}
          />
          <ExpressionQueue
            page={currentPage}
            selectedIds={selectedIds}
            onSelect={(id, multi) => setSelectedIds(multi ? Array.from(new Set([...selectedIds, id])) : [id])}
            onReorder={async (ids) => {
              if (!currentPage) return;
              await reorderExpressions(currentPage.id, ids);
              await refreshPage(currentPage.id);
            }}
          />
        </section>

        <Inspector
          expression={selectedExpression}
          selectedCount={selectedIds.length}
          onAccept={() => void setStatus("accepted")}
          onReject={() => void setStatus("rejected")}
          onSplit={handleSplit}
          onMerge={handleMerge}
          onNormalize={async (id) => {
            setBusy(true);
            await normalizeExpression(id);
            await refreshPage();
            setMessage("Đã tạo M4-ready crop nền đen chữ trắng.");
            setBusy(false);
          }}
          onRecognize={async (id) => {
            setBusy(true);
            try {
              await recognizeExpression(id);
              await refreshPage();
              setMessage("Đã chạy M4 bằng ảnh M4-ready và cập nhật LaTeX.");
            } catch (error) {
              const detail = typeof error === "object" && error && "response" in error ? (error as { response?: { data?: { detail?: string } } }).response?.data?.detail : undefined;
              setMessage(detail || "Không chạy được M4. Hãy kiểm tra M4-ready crop và M4 server.");
            } finally {
              setBusy(false);
            }
          }}
          onSaveLatex={async (id, latex) => {
            setBusy(true);
            await updateLatex(id, latex);
            await refreshPage();
            setMessage("Đã lưu LaTeX thủ công.");
            setBusy(false);
          }}
        />
      </main>
      <footer className="statusbar">{isBusy ? "Đang xử lý..." : message}</footer>
    </div>
  );
}

function Toolbar(props: {
  tool: Tool;
  layer: LayerKey;
  selectedCount: number;
  onTool: (tool: Tool) => void;
  onLayer: (layer: LayerKey) => void;
  onAccept: () => void;
  onReject: () => void;
  onDelete: () => void;
  onClearQueue: () => void;
  onMerge: () => void;
  onSplit: (mode: "horizontal" | "vertical") => void;
}) {
  return (
    <div className="toolbar">
      <button className={props.tool === "select" ? "active" : ""} onClick={() => props.onTool("select")} title="Select">
        <MousePointer2 size={16} /> Select
      </button>
      <button className={props.tool === "pan" ? "active" : ""} onClick={() => props.onTool("pan")} title="Pan">
        <Hand size={16} /> Pan
      </button>
      <button className={props.tool === "draw" ? "active" : ""} onClick={() => props.onTool("draw")} title="Draw Box">
        <BoxSelect size={16} /> Draw Box
      </button>
      <button onClick={() => props.onSplit("horizontal")} disabled={props.selectedCount !== 1}>
        <Scissors size={16} /> Split H
      </button>
      <button onClick={() => props.onSplit("vertical")} disabled={props.selectedCount !== 1}>
        <Scissors size={16} /> Split V
      </button>
      <button onClick={props.onMerge} disabled={props.selectedCount < 2}>
        <ChevronsUpDown size={16} /> Merge
      </button>
      <button onClick={props.onReject} disabled={!props.selectedCount}>
        <X size={16} /> Reject
      </button>
      <button onClick={props.onDelete} disabled={!props.selectedCount}>
        <Trash2 size={16} /> Delete
      </button>
      <button onClick={props.onClearQueue}>
        <Trash2 size={16} /> Clear Queue
      </button>
      <button onClick={props.onAccept} disabled={!props.selectedCount}>
        <Check size={16} /> Accept
      </button>
      <span className="toolbar-divider" />
      {(["original", "cleaned", "binary", "components", "normalized", "m4_ready"] as LayerKey[]).map((key) => (
        <button key={key} className={props.layer === key ? "active" : ""} onClick={() => props.onLayer(key)}>
          <Layers size={16} /> {key}
        </button>
      ))}
    </div>
  );
}

function PageViewer(props: {
  page?: PageItem;
  selectedIds: string[];
  tool: Tool;
  layer: LayerKey;
  onSelect: (ids: string[]) => void;
  onCreate: (bbox: BBox) => Promise<void>;
  onPatch: (id: string, payload: { bbox?: BBox; status?: ExpressionStatus }) => Promise<void>;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const rectRefs = useRef<Record<string, Konva.Rect | null>>({});
  const [size, setSize] = useState({ width: 900, height: 640 });
  const [scale, setScale] = useState(1);
  const [pos, setPos] = useState({ x: 20, y: 20 });
  const [draft, setDraft] = useState<BBox | null>(null);
  const imageUrl = props.page?.layers?.[props.layer] ?? props.page?.imageUrl;
  const image = useImage(imageUrl);

  useEffect(() => {
    const resize = () => {
      const rect = wrapRef.current?.getBoundingClientRect();
      if (rect) setSize({ width: rect.width, height: rect.height });
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  useEffect(() => {
    if (!props.page || !size.width) return;
    const nextScale = Math.min((size.width - 48) / props.page.width, (size.height - 48) / props.page.height, 1);
    setScale(nextScale);
    setPos({ x: Math.max(24, (size.width - props.page.width * nextScale) / 2), y: 24 });
  }, [props.page?.id, size.width, size.height]);

  useEffect(() => {
    const nodes = props.selectedIds.map((id) => rectRefs.current[id]).filter(Boolean) as Konva.Rect[];
    transformerRef.current?.nodes(nodes);
    transformerRef.current?.getLayer()?.batchDraw();
  }, [props.selectedIds, props.page?.expressions]);

  const stageToImage = () => {
    const stage = stageRef.current;
    const pointer = stage?.getPointerPosition();
    if (!pointer) return null;
    return { x: (pointer.x - pos.x) / scale, y: (pointer.y - pos.y) / scale };
  };

  const onWheel = (event: Konva.KonvaEventObject<WheelEvent>) => {
    event.evt.preventDefault();
    const direction = event.evt.deltaY > 0 ? -1 : 1;
    const factor = direction > 0 ? 1.08 : 0.92;
    setScale((value) => Math.max(0.08, Math.min(5, value * factor)));
  };

  if (!props.page) {
    return <div className="viewer empty-viewer">Upload ảnh để xem trang và chỉnh bbox.</div>;
  }
  const page = props.page;

  return (
    <div className="viewer" ref={wrapRef}>
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        x={props.tool === "pan" ? pos.x : 0}
        y={props.tool === "pan" ? pos.y : 0}
        draggable={props.tool === "pan"}
        onDragEnd={(event) => props.tool === "pan" && setPos({ x: event.target.x(), y: event.target.y() })}
        onWheel={onWheel}
        onMouseDown={(event) => {
          if (props.tool !== "draw") {
            if (event.target === event.target.getStage()) props.onSelect([]);
            return;
          }
          const point = stageToImage();
          if (point) setDraft({ x: point.x, y: point.y, width: 0, height: 0 });
        }}
        onMouseMove={() => {
          if (!draft || props.tool !== "draw") return;
          const point = stageToImage();
          if (point) setDraft({ ...draft, width: point.x - draft.x, height: point.y - draft.y });
        }}
        onMouseUp={() => {
          if (!draft || props.tool !== "draw") return;
          const bbox = clampBBox(draft, page.width, page.height);
          setDraft(null);
          if (bbox.width > 8 && bbox.height > 8) void props.onCreate(bbox);
        }}
      >
        <Layer x={props.tool === "pan" ? 0 : pos.x} y={props.tool === "pan" ? 0 : pos.y} scaleX={scale} scaleY={scale}>
          {image && <KonvaImage image={image} width={page.width} height={page.height} />}
          {page.expressions.map((expr) => (
            <React.Fragment key={expr.id}>
              <Rect
                ref={(node) => {
                  rectRefs.current[expr.id] = node;
                }}
                x={expr.bbox.x}
                y={expr.bbox.y}
                width={expr.bbox.width}
                height={expr.bbox.height}
                stroke={props.selectedIds.includes(expr.id) ? "#0284C7" : boxColor[expr.status]}
                strokeWidth={props.selectedIds.includes(expr.id) ? 4 / scale : 2 / scale}
                dash={expr.status === "rejected" ? [8 / scale, 6 / scale] : undefined}
                fill={props.selectedIds.includes(expr.id) ? "rgba(2,132,199,0.08)" : "rgba(37,99,235,0.04)"}
                draggable={props.tool === "select"}
                onClick={(event) => {
                  event.cancelBubble = true;
                  props.onSelect(event.evt.shiftKey ? Array.from(new Set([...props.selectedIds, expr.id])) : [expr.id]);
                }}
                onDblClick={() => window.open(assetUrl(expr.cropPreviewUrl), "_blank")}
                onDragEnd={(event) => {
                  const bbox = clampBBox({ ...expr.bbox, x: event.target.x(), y: event.target.y() }, page.width, page.height);
                  void props.onPatch(expr.id, { bbox, status: "edited" });
                }}
                onTransformEnd={(event) => {
                  const node = event.target;
                  const bbox = clampBBox(
                    normalizeBBox({
                      x: node.x(),
                      y: node.y(),
                      width: Math.max(4, node.width() * node.scaleX()),
                      height: Math.max(4, node.height() * node.scaleY())
                    }),
                    page.width,
                    page.height
                  );
                  node.scaleX(1);
                  node.scaleY(1);
                  void props.onPatch(expr.id, { bbox, status: "edited" });
                }}
              />
              <Text
                x={expr.bbox.x}
                y={Math.max(0, expr.bbox.y - 20 / scale)}
                text={`${expr.id} · ${statusLabel[expr.status]}`}
                fontSize={13 / scale}
                fill={props.selectedIds.includes(expr.id) ? "#0284C7" : boxColor[expr.status]}
                listening={false}
              />
            </React.Fragment>
          ))}
          {draft && (
            <Rect
              {...normalizeBBox(draft)}
              stroke="#0284C7"
              strokeWidth={2 / scale}
              dash={[6 / scale, 5 / scale]}
              fill="rgba(2,132,199,0.08)"
            />
          )}
          <Transformer ref={transformerRef} rotateEnabled={false} boundBoxFunc={(_, next) => (next.width < 8 || next.height < 8 ? _ : next)} />
        </Layer>
      </Stage>
      <button className="fit-button" onClick={() => {
        const nextScale = Math.min((size.width - 48) / page.width, (size.height - 48) / page.height, 1);
        setScale(nextScale);
        setPos({ x: Math.max(24, (size.width - page.width * nextScale) / 2), y: 24 });
      }}>
        <Maximize size={16} /> Fit Page
      </button>
      <div className="zoom-indicator">{Math.round(scale * 100)}%</div>
    </div>
  );
}

function Inspector(props: {
  expression?: ExpressionBox;
  selectedCount: number;
  onAccept: () => void;
  onReject: () => void;
  onSplit: (mode: "horizontal" | "vertical") => void;
  onMerge: () => void;
  onNormalize: (id: string) => Promise<void>;
  onRecognize: (id: string) => Promise<void>;
  onSaveLatex: (id: string, latex: string) => Promise<void>;
}) {
  const expr = props.expression;
  const [latexDraft, setLatexDraft] = useState("");
  useEffect(() => {
    setLatexDraft(expr?.latexClean ?? expr?.latexRaw ?? "");
  }, [expr?.id, expr?.latexClean, expr?.latexRaw]);
  const renderedLatex = useMemo(() => {
    if (!latexDraft.trim()) return "";
    try {
      return katex.renderToString(latexDraft, { throwOnError: false, displayMode: true });
    } catch {
      return "";
    }
  }, [latexDraft]);
  const latestRecognitionError = useMemo(() => {
    const history = expr?.recognitionHistory ?? [];
    const last = history[history.length - 1];
    return typeof last?.error === "string" ? last.error : "";
  }, [expr?.recognitionHistory]);
  const canRunM4 =
    !!expr &&
    !!(expr.normalizedUrl ?? expr.normalizedPreviewUrl) &&
    expr.candidateType === "single_expression" &&
    !["rejected", "noise", "fragment"].includes(expr.status) &&
    !expr.quality.emptyAfterNormalize;
  const m4BlockReason = !expr
    ? ""
    : !(expr.normalizedUrl ?? expr.normalizedPreviewUrl)
      ? "Chưa có M4-ready crop. Hãy bấm Normalize Preview trước."
      : expr.candidateType !== "single_expression"
        ? "M4 chỉ nhận single_expression. Hãy split/merge/chỉnh bbox trước."
        : ["rejected", "noise", "fragment"].includes(expr.status)
          ? "Expression đang bị rejected/noise/fragment."
          : expr.quality.emptyAfterNormalize
            ? "Ảnh normalize rỗng hoặc lỗi. Hãy chỉnh bbox rồi normalize lại."
            : "";
  return (
    <aside className="inspector">
      <div className="panel-title">
        <span>Inspector</span>
        <small>{props.selectedCount ? `${props.selectedCount} bbox đang chọn` : "Chưa chọn bbox"}</small>
      </div>
      {!expr ? (
        <div className="empty">Chọn một bbox để xem crop preview, quality và lịch sử chỉnh sửa.</div>
      ) : (
        <>
          <section>
            <h2>Selected: {expr.id}</h2>
            <span className={`badge ${statusClass[expr.status]}`}>{statusLabel[expr.status]}</span>
            <span className={`badge candidate-${expr.candidateType}`}>{candidateLabel[expr.candidateType]}</span>
          </section>
          {expr.candidateType === "multiline_block" && (
            <section className="warning-callout">
              Block này có vẻ chứa nhiều dòng/biểu thức. M4 chỉ nhận một biểu thức/lần. Hãy dùng Split ngang trước khi Run M4.
            </section>
          )}
          <section>
            <h3>Preview</h3>
            <div className="preview-grid">
              <PreviewImage label="Original crop" url={expr.cropPreviewUrl} version={`${expr.id}-${expr.updatedAt}-original`} />
              <PreviewImage label="Cleaned crop" url={expr.cleanedPreviewUrl} version={`${expr.id}-${expr.updatedAt}-cleaned`} />
              <PreviewImage label="Binary crop" url={expr.binaryPreviewUrl} version={`${expr.id}-${expr.updatedAt}-binary`} />
              <PreviewImage label="M4-ready crop" url={expr.normalizedPreviewUrl ?? expr.normalizedUrl} version={`${expr.id}-${expr.updatedAt}-m4`} m4 />
            </div>
          </section>
          <section className="quality-grid">
            <h3>Quality</h3>
            <span>Width</span><strong>{Math.round(expr.bbox.width)} px</strong>
            <span>Height</span><strong>{Math.round(expr.bbox.height)} px</strong>
            <span>Aspect ratio</span><strong>{expr.quality.aspectRatio ?? "N/A"}</strong>
            <span>Foreground</span><strong>{expr.quality.foregroundRatio != null ? `${(expr.quality.foregroundRatio * 100).toFixed(1)}%` : "N/A"}</strong>
            <span>Touch border</span><strong>{expr.quality.touchBorder ? "Có" : "Không"}</strong>
            <span>Multiple</span><strong>{expr.quality.maybeMultipleExpressions ? "Có thể" : "Không rõ"}</strong>
            <span>Components</span><strong>{expr.quality.componentCount ?? "N/A"}</strong>
          </section>
          <section>
            <h3>Warnings</h3>
            <div className="warning-list">
              {[...new Set([...(expr.quality.warnings ?? []), ...(expr.warnings ?? [])])].length
                ? [...new Set([...(expr.quality.warnings ?? []), ...(expr.warnings ?? [])])].map((warn) => <span key={warn}>{warn}</span>)
                : <em>Không có cảnh báo.</em>}
            </div>
          </section>
          <section className="action-grid">
            <button className="button primary" onClick={props.onAccept}><Check size={16} /> Accept</button>
            <button className="button" onClick={props.onReject}><X size={16} /> Reject</button>
            <button className="button" onClick={() => props.onSplit("horizontal")}><Scissors size={16} /> Split ngang</button>
            <button className="button" onClick={() => props.onSplit("vertical")}><Scissors size={16} /> Split dọc</button>
            <button className="button" onClick={props.onMerge} disabled={props.selectedCount < 2}><ChevronsUpDown size={16} /> Merge</button>
            <button className="button" onClick={() => void props.onNormalize(expr.id)}><FileJson size={16} /> 1. Normalize Preview</button>
            <button className="button primary" onClick={() => void props.onRecognize(expr.id)} disabled={!canRunM4} title={m4BlockReason}><ScanLine size={16} /> 2. Run M4</button>
          </section>
          {m4BlockReason && <section className="m4-gate-note">{m4BlockReason}</section>}
          <section className="recognition-panel">
            <h3>Recognition</h3>
            <div className="quality-grid">
              <span>Status</span><strong>{expr.latexStatus ?? "not_run"}</strong>
              <span>Confidence</span><strong>{expr.latexConfidence != null ? `${Math.round(expr.latexConfidence * 100)}%` : "N/A"}</strong>
            </div>
            {expr.latexStatus === "model_error" && latestRecognitionError && (
              <div className="warning-callout">Lỗi M4: {latestRecognitionError}</div>
            )}
            <label className="latex-label">Latex Raw</label>
            <pre className="latex-raw">{expr.latexRaw || "Chưa chạy M4."}</pre>
            <label className="latex-label">Latex Clean/Edit</label>
            <textarea className="latex-editor" value={latexDraft} onChange={(event) => setLatexDraft(event.target.value)} placeholder="Nhập hoặc sửa LaTeX tại đây" />
            <div className="latex-actions">
              <button className="button" onClick={() => void props.onSaveLatex(expr.id, latexDraft)} disabled={!latexDraft.trim()}>Save LaTeX</button>
              <button className="button" onClick={() => void navigator.clipboard.writeText(latexDraft)} disabled={!latexDraft.trim()}>Copy LaTeX</button>
              <button className="button" onClick={() => void navigator.clipboard.writeText(`$${latexDraft}$`)} disabled={!latexDraft.trim()}>Copy Markdown</button>
            </div>
            <div className="latex-render" dangerouslySetInnerHTML={{ __html: renderedLatex || "<span>Chưa có LaTeX để render.</span>" }} />
          </section>
          <section>
            <h3>History</h3>
            <ul className="history">
              {expr.history.slice(-5).reverse().map((item, index) => <li key={`${item.at}-${index}`}>{item.action} · {item.by}</li>)}
            </ul>
          </section>
        </>
      )}
    </aside>
  );
}

function cacheBustedUrl(url?: string, version?: string) {
  if (!url) return "";
  const separator = url.includes("?") ? "&" : "?";
  return `${assetUrl(url)}${separator}v=${encodeURIComponent(version ?? "")}`;
}

function PreviewImage(props: { label: string; url?: string; version?: string; m4?: boolean }) {
  const src = cacheBustedUrl(props.url, props.version);
  return (
    <div className={props.m4 ? "preview-card preview-card-m4" : "preview-card"}>
      <span>{props.label}</span>
      {props.url ? <img key={src} src={src} alt={props.label} /> : <em>Chưa có</em>}
    </div>
  );
}

function ExpressionQueue(props: {
  page?: PageItem;
  selectedIds: string[];
  onSelect: (id: string, multi: boolean) => void;
  onReorder: (ids: string[]) => Promise<void>;
}) {
  const [dragId, setDragId] = useState<string>();
  const expressions = props.page?.expressions.slice().sort((a, b) => a.order - b.order) ?? [];
  return (
    <div className="queue">
      <div className="queue-title">
        <strong>Expression Queue</strong>
        <span>Thứ tự đọc: từ trên xuống dưới, từ trái sang phải. Có thể kéo thả để đổi thứ tự.</span>
      </div>
      <div className="queue-list">
        {expressions.length === 0 ? (
          <div className="empty queue-empty">Chưa có expression. Hãy Auto Scan hoặc Draw Box.</div>
        ) : (
          expressions.map((expr) => (
            <button
              key={expr.id}
              draggable
              className={`queue-card ${props.selectedIds.includes(expr.id) ? "active" : ""}`}
              onClick={(event) => props.onSelect(expr.id, event.shiftKey)}
              onDragStart={() => setDragId(expr.id)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={() => {
                if (!dragId || dragId === expr.id) return;
                const ids = expressions.map((item) => item.id);
                const from = ids.indexOf(dragId);
                const to = ids.indexOf(expr.id);
                ids.splice(to, 0, ids.splice(from, 1)[0]);
                void props.onReorder(ids);
                setDragId(undefined);
              }}
            >
              <span>#{expr.order}</span>
              {expr.cropPreviewUrl && <img src={assetUrl(expr.cropPreviewUrl)} alt={expr.id} />}
              <strong>{expr.id}</strong>
              <em className={`badge ${statusClass[expr.status]}`}>{statusLabel[expr.status]}</em>
              <em className={`badge candidate-${expr.candidateType}`}>{candidateLabel[expr.candidateType]}</em>
              {expr.quality.warnings.length > 0 && <small>{expr.quality.warnings.length} cảnh báo</small>}
            </button>
          ))
        )}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
