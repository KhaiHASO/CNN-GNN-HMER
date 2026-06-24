import type { BBox } from "./types";

export function normalizeBBox(box: BBox): BBox {
  const x = box.width < 0 ? box.x + box.width : box.x;
  const y = box.height < 0 ? box.y + box.height : box.y;
  return { x, y, width: Math.abs(box.width), height: Math.abs(box.height) };
}

export function clampBBox(box: BBox, width: number, height: number): BBox {
  const normalized = normalizeBBox(box);
  const x = Math.max(0, Math.min(width, normalized.x));
  const y = Math.max(0, Math.min(height, normalized.y));
  return {
    x,
    y,
    width: Math.max(1, Math.min(width - x, normalized.width)),
    height: Math.max(1, Math.min(height - y, normalized.height))
  };
}

export function unionBBoxes(boxes: BBox[]): BBox {
  const x1 = Math.min(...boxes.map((b) => b.x));
  const y1 = Math.min(...boxes.map((b) => b.y));
  const x2 = Math.max(...boxes.map((b) => b.x + b.width));
  const y2 = Math.max(...boxes.map((b) => b.y + b.height));
  return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
}

