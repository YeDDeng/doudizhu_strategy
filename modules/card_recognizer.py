"""
Card Recognizer - Pure YOLO Detection
Uses YOLO model to detect and classify cards directly.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO

# Lazy cv2 import (Python 3.14 compat)
_cv2 = None
def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _cv2_module
        _cv2 = _cv2_module
    return _cv2


class CardRecognizer:
    """
    Pure YOLO card recognition - no template matching.
    """

    # Class ID to rank name mapping
    CLASS_MAP = {
        0: '3', 1: '4', 2: '5', 3: '6', 4: '7', 5: '8', 6: '9', 7: '10',
        8: 'J', 9: 'Q', 10: 'K', 11: 'A', 12: '2', 13: 'Joker_B', 14: 'Joker_R'
    }

    def __init__(self, model_path: str = "models/yolov8_cards.pt",
                 confidence_threshold: float = 0.15):
        """
        Initialize recognizer.

        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Detection confidence threshold
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None
        self._load_model()
        # Temporal smoothing: keep last N frames for stable detection
        self._my_cards_history = []
        self._history_maxlen = 5  # Keep last 5 frames
        # 3-frame debounce for history reset (prevents single-frame misdetection from clearing history)
        self._card_empty_frames = 0
        # Stability tracking: only report when detection actually changes
        self._last_reported_cards: Optional[List[str]] = None
        self._last_reported_upper: Optional[List[str]] = None
        self._last_reported_lower: Optional[List[str]] = None

    def reset_history(self) -> None:
        """Reset temporal smoothing history."""
        self._my_cards_history = []

    def _load_model(self) -> None:
        """Load YOLOv8 model(s)."""
        try:
            self.model = YOLO(self.model_path)
            print(f"[CardRecognizer] Loaded YOLO model from {self.model_path}")
        except Exception as e:
            print(f"[CardRecognizer] Failed to load model: {e}")
            self.model = None
            self.model_center = None

    def _preprocess_for_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for YOLO detection.
        Returns preprocessed image and scale factor used.
        """
        if image is None or image.size == 0:
            return image, 1.0

        h, w = image.shape[:2]

        # Scale up small images to improve detection
        scale = 1.0
        if w < 800 or h < 600:
            scale = max(800 / w, 600 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = _get_cv2().resize(image, (new_w, new_h))

        # Check if image is dark (needs enhancement)
        is_dark = False
        if len(image.shape) == 3:
            gray = _get_cv2().cvtColor(image, _get_cv2().COLOR_BGR2GRAY)
            mean_brightness = gray.mean()
            is_dark = mean_brightness < 80  # Only enhance dark images
        else:
            mean_brightness = image.mean()
            is_dark = mean_brightness < 80

        # Only apply CLAHE preprocessing for dark images
        if is_dark and len(image.shape) == 3:
            clahe = _get_cv2().createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            bright = _get_cv2().convertScaleAbs(enhanced, alpha=1.5, beta=30)
            processed = _get_cv2().cvtColor(bright, _get_cv2().COLOR_GRAY2BGR)
        elif is_dark:
            clahe = _get_cv2().createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            processed = _get_cv2().convertScaleAbs(enhanced, alpha=1.5, beta=30)
        else:
            # No preprocessing needed for well-lit images
            processed = image

        return processed, scale

    def _get_yolo_detections(self, image: np.ndarray, conf_override: float = None,
                              model: Optional[YOLO] = None) -> List[Dict]:
        """Get card bounding boxes from YOLO.

        Args:
            image: Input image
            conf_override: Override confidence threshold (for opponent cards)
            model: YOLO model instance (defaults to self.model)
        """
        model = model or self.model
        if model is None:
            return []

        conf = conf_override if conf_override is not None else self.confidence_threshold
        imgsz = 480  # Reduced from 640 for faster CPU inference (model was trained at 640)

        # Apply CLAHE enhancement for dark images only (no scaling — YOLO handles resizing via imgsz)
        h, w = image.shape[:2]
        processed = image
        if h > 0 and w > 0 and len(image.shape) == 3:
            gray = _get_cv2().cvtColor(image, _get_cv2().COLOR_BGR2GRAY)
            if gray.mean() < 80:  # Dark image
                clahe = _get_cv2().createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                bright = _get_cv2().convertScaleAbs(enhanced, alpha=1.5, beta=30)
                processed = _get_cv2().cvtColor(bright, _get_cv2().COLOR_GRAY2BGR)

        results = model(processed, conf=conf, iou=0.4, imgsz=imgsz, verbose=False)
        scale = 1.0  # No scaling applied

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0]
                cls = int(box.cls[0])

                # Scale coordinates back to original image space
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

                rank = self.CLASS_MAP.get(cls, '?')

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'center_x': int((x1 + x2) / 2),
                    'center_y': int((y1 + y2) / 2),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'card': rank,
                    'confidence': float(conf),
                    'class_id': cls
                })

        return detections

    def _cluster_detections(self, detections: List[Dict]) -> List[Dict]:
        """Cluster overlapping detections using IoU-based NMS."""
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)

        keep = []
        suppressed = set()

        for i, det in enumerate(detections):
            if i in suppressed:
                continue

            keep.append(det)
            x1_i, y1_i, x2_i, y2_i = det['bbox']
            card_i = det['card']

            for j, other in enumerate(detections[i+1:], i+1):
                if j in suppressed:
                    continue

                x1_j, y1_j, x2_j, y2_j = other['bbox']
                card_j = other['card']

                # Calculate IoU
                inter_x1 = max(x1_i, x1_j)
                inter_y1 = max(y1_i, y1_j)
                inter_x2 = min(x2_i, x2_j)
                inter_y2 = min(y2_i, y2_j)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area_i = (x2_i - x1_i) * (y2_i - y1_i)
                    area_j = (x2_j - x1_j) * (y2_j - y1_j)
                    union_area = area_i + area_j - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    # Same class + significant overlap -> suppress lower confidence
                    if card_i == card_j and iou > 0.35:
                        suppressed.add(j)

        return keep

    def _deduplicate_by_position(self, detections: List[Dict]) -> List[Dict]:
        """Deduplicate detections at same horizontal position (within 40px)."""
        if not detections:
            return []

        # Group by approximate x position (within 40px = roughly half card width)
        groups = {}
        for det in detections:
            x_key = det['center_x'] // 40
            if x_key not in groups:
                groups[x_key] = []
            groups[x_key].append(det)

        # Keep highest confidence per position
        filtered = []
        for x_key, group in groups.items():
            # Same card type: keep highest confidence
            same_type = {}
            for det in group:
                card = det['card']
                if card not in same_type:
                    same_type[card] = det
                elif det['confidence'] > same_type[card]['confidence']:
                    same_type[card] = det
            filtered.extend(same_type.values())

        return filtered

    def _filter_by_frequency(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections based on expected card frequency."""
        if len(detections) <= 1:
            return detections

        max_per_rank = {'Joker_B': 1, 'Joker_R': 1}

        card_groups = {}
        for det in detections:
            card = det['card']
            if card not in card_groups:
                card_groups[card] = []
            card_groups[card].append(det)

        filtered = []
        for card, group in card_groups.items():
            group.sort(key=lambda d: d['confidence'], reverse=True)
            max_allowed = max_per_rank.get(card, 4)
            filtered.extend(group[:max_allowed])

        return filtered

    def _filter_by_bbox_shape(self, detections: List[Dict]) -> List[Dict]:
        """Filter out detections with abnormal bbox aspect ratios."""
        valid = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            ratio = w / h
            # Card bbox should be roughly rectangular, not too flat or too tall
            if 0.2 <= ratio <= 5.0:
                valid.append(det)
        return valid

    def _filter_q10_confusion(self, detections: List[Dict]) -> List[Dict]:
        """
        Fix Q/10 confusion using two strategies:
        1. Nearby Q/10 pairs: within 30px, keep higher confidence
        2. Cluster-based: if many Qs (>=3) but few 10s (<=1),
           and 10 is nearby (within 150px), suppress lower-conf Qs near it
        """
        if not detections:
            return detections

        # Group Q and 10 detections by approximate x position
        q_dets = [(i, d) for i, d in enumerate(detections) if d['card'] == 'Q']
        ten_dets = [(i, d) for i, d in enumerate(detections) if d['card'] == '10']

        if not q_dets or not ten_dets:
            return detections

        suppress_q = set()
        suppress_ten = set()

        # Strategy 1: Close Q/10 pairs (within 30px)
        for q_idx, q_det in q_dets:
            if q_idx in suppress_q:
                continue
            q_x = q_det['center_x']
            q_conf = q_det['confidence']

            for ten_idx, ten_det in ten_dets:
                if ten_idx in suppress_ten:
                    continue
                ten_x = ten_det['center_x']
                ten_conf = ten_det['confidence']

                if abs(q_x - ten_x) <= 30:
                    if q_conf > ten_conf * 1.3:
                        suppress_ten.add(ten_idx)
                    elif ten_conf > q_conf * 1.3:
                        suppress_q.add(q_idx)

        # Strategy 2: Cluster-based (many Qs but few 10s)
        if len(q_dets) >= 3 and len(ten_dets) <= 1:
            # Find the 10 with highest confidence
            best_ten = max(ten_dets, key=lambda x: x[1]['confidence']) if ten_dets else None
            if best_ten:
                ten_x = best_ten[1]['center_x']
                ten_conf = best_ten[1]['confidence']

                # For each Q within 150px of this 10
                for q_idx, q_det in q_dets:
                    if q_idx in suppress_q:
                        continue
                    q_x = q_det['center_x']
                    q_conf = q_det['confidence']

                    if abs(q_x - ten_x) <= 150:
                        # If 10 has higher confidence, suppress Q
                        if ten_conf > q_conf:
                            suppress_q.add(q_idx)

        # Build filtered list
        filtered = []
        for i, det in enumerate(detections):
            if det['card'] == 'Q' and i in suppress_q:
                continue
            if det['card'] == '10' and i in suppress_ten:
                continue
            filtered.append(det)

        return filtered

    def _supplement_jokers(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        Supplement missing Joker detections using lower threshold and color verification.
        Only supplements when ZERO of either type are detected, and requires reasonable confidence.
        """
        if not detections:
            return detections

        # Count current jokers (only 1 of each exists in the deck)
        joker_r_count = sum(1 for d in detections if d['card'] == 'Joker_R')
        joker_b_count = sum(1 for d in detections if d['card'] == 'Joker_B')

        # Only supplement if we're missing at least one type entirely
        if joker_r_count >= 1 and joker_b_count >= 1:
            return detections

        # Get existing joker positions to avoid duplicates
        existing_joker_positions = {}
        for d in detections:
            if d['card'] == 'Joker_R':
                existing_joker_positions['Joker_R'] = d['center_x'] // 50
            elif d['card'] == 'Joker_B':
                existing_joker_positions['Joker_B'] = d['center_x'] // 50

        # Run additional detection with lower threshold
        if self.model is None:
            return detections

        # Preprocess image
        processed, scale = self._preprocess_for_yolo(image)
        if processed is None:
            return detections

        # Detect with moderate threshold — 480 for consistent speed
        results = self.model(processed, conf=0.15, iou=0.3, imgsz=480, verbose=False)

        new_jokers = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                rank = self.CLASS_MAP.get(cls, '?')
                if rank not in ('Joker_R', 'Joker_B'):
                    continue

                # Scale coordinates back
                x1 = int(x1 / scale)
                x2 = int(x2 / scale)
                y1 = int(y1 / scale)
                y2 = int(y2 / scale)

                center_x = int((x1 + x2) / 2)

                # Already have this joker type
                if rank == 'Joker_R' and joker_r_count >= 1:
                    continue
                if rank == 'Joker_B' and joker_b_count >= 1:
                    continue

                # Skip if this position is already covered (within 100px)
                pos_key = center_x // 50
                if rank in existing_joker_positions:
                    if abs(center_x - existing_joker_positions[rank] * 50) < 100:
                        continue

                # Skip if confidence is too low
                if conf < 0.15:
                    continue

                new_jokers.append({
                    'bbox': (x1, y1, x2, y2),
                    'center_x': center_x,
                    'center_y': int((y1 + y2) / 2),
                    'card': rank,
                    'confidence': conf,
                    'class_id': cls
                })

        # Add new jokers to detections
        if new_jokers:
            detections = detections + new_jokers

        return detections

    def _supplement_low_confidence_cards(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        Supplement missing cards (2, 8, K, etc.) using very low threshold when detection count is low.
        Only for card_type='my' when we detect fewer than 16 cards.
        """
        if not detections or image is None:
            return detections

        if self.model is None:
            return detections

        # Check if we already have these cards
        detected_cards = set(d['card'] for d in detections)
        missing_important = []
        for card in ['2', '8', 'K', 'Q']:
            if card not in detected_cards:
                missing_important.append(card)

        if not missing_important:
            return detections

        # Get existing positions to avoid duplicates
        existing_positions = set()
        for d in detections:
            existing_positions.add(d['center_x'] // 40)

        # Run detection with very low threshold
        processed, scale = self._preprocess_for_yolo(image)
        if processed is None:
            return detections

        results = self.model(processed, conf=0.015, iou=0.3, imgsz=480, verbose=False)

        new_cards = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                rank = self.CLASS_MAP.get(cls, '?')

                # Only interested in missing important cards
                if rank not in missing_important:
                    continue

                # Scale coordinates back
                x1 = int(x1 / scale)
                x2 = int(x2 / scale)
                y1 = int(y1 / scale)
                y2 = int(y2 / scale)

                center_x = int((x1 + x2) / 2)

                # Skip if position is already covered
                if center_x // 40 in existing_positions:
                    continue

                # Skip if confidence is too low
                if conf < 0.015:
                    continue

                new_cards.append({
                    'bbox': (x1, y1, x2, y2),
                    'center_x': center_x,
                    'center_y': int((y1 + y2) / 2),
                    'card': rank,
                    'confidence': conf,
                    'class_id': cls
                })

        if new_cards:
            # Only add cards that don't create duplicates
            for card in new_cards:
                pos_key = card['center_x'] // 40
                if pos_key not in existing_positions:
                    detections.append(card)
                    existing_positions.add(pos_key)

        return detections

    def _verify_and_correct_jokers_by_color(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        Verify Joker_R vs Joker_B classification using color analysis.
        Red regions -> Joker_R, Black regions -> Joker_B
        """
        if not detections or image is None:
            return detections

        # Count jokers
        joker_r_count = sum(1 for d in detections if d['card'] == 'Joker_R')
        joker_b_count = sum(1 for d in detections if d['card'] == 'Joker_B')

        # If we have both jokers correctly, no need to verify
        if joker_r_count >= 1 and joker_b_count >= 1:
            return detections

        # Get the original image (not preprocessed) for color analysis
        orig_h, orig_w = image.shape[:2] if len(image.shape) == 3 else (0, 0)
        if orig_h == 0:
            return detections

        # Analyze each joker detection
        for i, det in enumerate(detections):
            if det['card'] not in ('Joker_R', 'Joker_B'):
                continue

            x1, y1, x2, y2 = det['bbox']
            center_x = det['center_x']
            center_y = det['center_y']

            # Extract region for color analysis (slightly larger than bbox)
            margin = 10
            x1_crop = max(0, x1 - margin)
            y1_crop = max(0, y1 - margin)
            x2_crop = min(orig_w, x2 + margin)
            y2_crop = min(orig_h, y2 + margin)

            if x2_crop <= x1_crop or y2_crop <= y1_crop:
                continue

            roi = image[y1_crop:y2_crop, x1_crop:x2_crop]

            # Calculate average color
            if len(roi.shape) == 3:
                avg_color = roi.mean(axis=(0, 1))  # B, G, R order in OpenCV
                b, g, r = avg_color[0], avg_color[1], avg_color[2]

                # Red Joker has higher R than B, Black Joker has higher B/G than R
                # Red ratio: R - max(B, G)
                red_ratio = r - max(b, g)

                # Determine joker type by color
                if red_ratio > 15:  # Red dominant -> Joker_R
                    detected_type = 'Joker_R'
                elif red_ratio < -15:  # Not red dominant -> likely Joker_B
                    detected_type = 'Joker_B'
                else:
                    detected_type = det['card']  # Keep original

                # If classification differs, correct it
                if detected_type != det['card']:
                    # Update joker count tracking
                    old_type = det['card']
                    detections[i]['card'] = detected_type
                    detections[i]['confidence'] = min(det['confidence'] + 0.1, 1.0)  # Boost confidence

        return detections

    def _validate_card_count(self, cards: List[str], max_cards: int = 20) -> List[str]:
        """Validate total card count and apply emergency truncation if needed."""
        if len(cards) <= max_cards:
            return cards
        # If somehow we got more cards than possible, this indicates severe mis-detection
        # Return empty to trigger temporal smoothing fallback
        return []

    def detect_cards(self, image: np.ndarray, card_type: str = "my") -> List[Dict]:
        """
        Detect cards using pure YOLO.

        Args:
            image: Input image (BGR format)
            card_type: Type of cards to detect ("my", "opponent")

        Returns:
            List of detected cards with bounding boxes
        """
        if image is None or image.size == 0:
            return []

        # Step 1: Get raw detections from YOLO
        # For opponent cards, use lower threshold since they may be smaller/lower contrast
        if card_type == "opponent":
            detections = self._get_yolo_detections(image, conf_override=0.01)
        else:
            detections = self._get_yolo_detections(image)

        if not detections:
            return []

        # Step 2: Filter by bbox shape (exclude abnormal aspect ratios)
        detections = self._filter_by_bbox_shape(detections)

        # Step 3: IoU-based clustering to merge overlapping boxes
        detections = self._cluster_detections(detections)

        # Step 4: Position-based deduplication (fallback for non-overlapping duplicates)
        detections = self._deduplicate_by_position(detections)

        # Step 5: Frequency filter (max 4 of each rank, 1 for jokers)
        filtered = self._filter_by_frequency(detections)

        # Step 6: Fix Q/10 confusion
        filtered = self._filter_q10_confusion(filtered)

        # Step 6.5: Supplement low-confidence cards (2, 8, K, etc.) if count is low
        if card_type == "my" and len(filtered) < 15:
            filtered = self._supplement_low_confidence_cards(filtered, image)

        # Step 7: Supplement missing jokers with lower threshold
        if card_type == "my":
            filtered = self._supplement_jokers(filtered, image)

        # Step 8: Verify and correct joker classification using color
        filtered = self._verify_and_correct_jokers_by_color(filtered, image)

        # Step 9: Final position-based deduplication after joker supplement
        filtered = self._deduplicate_by_position(filtered)

        # Step 10: Sort by x position
        filtered.sort(key=lambda d: d['center_x'])

        # Format output
        results = [{
            'card': d['card'],
            'confidence': d['confidence'],
            'center_x': d['center_x'],
            'bbox': d['bbox']
        } for d in filtered]

        return results

    def _process_region_detections(self, detections: List[Dict], card_type: str,
                                    image: np.ndarray) -> List[Dict]:
        """Apply region-specific post-processing to detections (shared helper)."""
        if not detections:
            return []

        detections = self._filter_by_bbox_shape(detections)
        detections = self._cluster_detections(detections)
        detections = self._deduplicate_by_position(detections)
        detections = self._filter_by_frequency(detections)
        detections = self._filter_q10_confusion(detections)

        if card_type == "my":
            detections = self._verify_and_correct_jokers_by_color(detections, image)
            detections = self._deduplicate_by_position(detections)

        detections.sort(key=lambda d: d['center_x'])
        return detections

    def recognize(self, screenshot: np.ndarray, regions: Dict) -> Dict:
        """
        Per-region crop + sequential YOLO inference on each crop independently.
        Hand and center regions processed separately for maximum detection quality.
        """
        h, w = screenshot.shape[:2]
        result = {
            "my_cards": [],
            "upper_player_last": [],
            "lower_player_last": [],
            "upper_player_count": 0,
            "lower_player_count": 0,
            "landlord": None,
            "current_turn": None
        }

        hand_rect = regions.get("my_hand")
        upper_rect = regions.get("upper_player_area")
        lower_rect = regions.get("lower_player_area")

        # Crop both regions
        hand_crop = None
        if hand_rect:
            hx1 = int(hand_rect[0] * w); hy1 = int(hand_rect[1] * h)
            hx2 = int(hand_rect[2] * w); hy2 = int(hand_rect[3] * h)
            hand_crop = screenshot[hy1:hy2, hx1:hx2]

        center_crop = None
        if upper_rect or lower_rect:
            cy1 = h; cy2 = 0; cx1 = w; cx2 = 0
            if upper_rect:
                ux1 = int(upper_rect[0] * w); uy1 = int(upper_rect[1] * h)
                ux2 = int(upper_rect[2] * w); uy2 = int(upper_rect[3] * h)
                cy1 = min(cy1, uy1); cy2 = max(cy2, uy2)
                cx1 = min(cx1, ux1); cx2 = max(cx2, ux2)
            if lower_rect:
                lx1 = int(lower_rect[0] * w); ly1 = int(lower_rect[1] * h)
                lx2 = int(lower_rect[2] * w); ly2 = int(lower_rect[3] * h)
                cy1 = min(cy1, ly1); cy2 = max(cy2, ly2)
                cx1 = min(cx1, lx1); cx2 = max(cx2, lx2)
            center_crop = screenshot[cy1:cy2, cx1:cx2]

        # === Separate inferences: hand and center independently at full crop resolution ===
        raw_cards = []
        center_dets = []

        has_hand = hand_crop is not None and hand_crop.size > 0 and hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0
        has_center = center_crop is not None and center_crop.size > 0 and center_crop.shape[0] > 0 and center_crop.shape[1] > 0

        if has_hand:
            dets = self._get_yolo_detections(hand_crop)
            dets = self._process_region_detections(dets, "my", hand_crop)
            # Supplement missing cards when detection count is low
            if len(dets) < 15:
                dets = self._supplement_low_confidence_cards(dets, hand_crop)
            dets = self._supplement_jokers(dets, hand_crop)
            dets = self._deduplicate_by_position(dets)
            raw_cards = [d['card'] for d in dets]

        if has_center:
            dets = self._get_yolo_detections(center_crop, conf_override=0.05)
            dets = self._process_region_detections(dets, "opponent", center_crop)
            # Adjust center to full-frame for upper/lower split
            for d in dets:
                d['center_x'] += cx1
                d['center_y'] += cy1
            center_dets = dets

        # === Post-process hand results ===
        if hand_rect:
            # Round-end detection: 3-frame debounce
            if self._my_cards_history and len(self._my_cards_history[-1]) > 0 and len(raw_cards) == 0:
                self._card_empty_frames += 1
                if self._card_empty_frames >= 3:
                    self._my_cards_history = []
                    self._card_empty_frames = 0
            else:
                self._card_empty_frames = 0

            # Temporal smoothing
            self._my_cards_history.append(raw_cards)
            if len(self._my_cards_history) > self._history_maxlen:
                self._my_cards_history.pop(0)

            if len(self._my_cards_history) >= 3:
                from collections import Counter as _Counter
                all_card_counts = [len(c) for c in self._my_cards_history]
                count_counter = _Counter(all_card_counts)
                most_common_count, count = count_counter.most_common(1)[0]

                if count >= len(all_card_counts) * 0.6:
                    for entry in reversed(self._my_cards_history):
                        if len(entry) == most_common_count:
                            result["my_cards"] = list(entry)
                            break
                    if not result["my_cards"]:
                        result["my_cards"] = list(self._my_cards_history[-1])
                else:
                    prev = self._my_cards_history[-2] if len(self._my_cards_history) > 1 else raw_cards
                    if prev and len(prev) >= 10 and len(raw_cards) < len(prev) - 5:
                        result["my_cards"] = list(prev)
                    else:
                        result["my_cards"] = list(self._my_cards_history[-1])
            else:
                result["my_cards"] = raw_cards

            result["my_cards"] = self._validate_card_count(result["my_cards"])

        # === Post-process center results (split into upper/lower) ===
        if center_dets:
            if upper_rect:
                ux1 = int(upper_rect[0] * w); uy1 = int(upper_rect[1] * h)
                ux2 = int(upper_rect[2] * w); uy2 = int(upper_rect[3] * h)
                result["upper_player_last"] = [
                    d['card'] for d in center_dets
                    if ux1 <= d['center_x'] <= ux2 and uy1 <= d['center_y'] <= uy2
                ]
            if lower_rect:
                lx1 = int(lower_rect[0] * w); ly1 = int(lower_rect[1] * h)
                lx2 = int(lower_rect[2] * w); ly2 = int(lower_rect[3] * h)
                result["lower_player_last"] = [
                    d['card'] for d in center_dets
                    if lx1 <= d['center_x'] <= lx2 and ly1 <= d['center_y'] <= ly2
                ]

        # === Stability: only report when detection meaningfully changed ===
        # This prevents AI suggestions from flipping every frame due to detection noise
        def _cards_changed(new_cards: List[str], last_cards: Optional[List[str]]) -> bool:
            if last_cards is None:
                return True
            if len(new_cards) != len(last_cards):
                return True
            return sorted(new_cards) != sorted(last_cards)

        current_my = result.get("my_cards", [])
        current_upper = result.get("upper_player_last", [])
        current_lower = result.get("lower_player_last", [])

        if (_cards_changed(current_my, self._last_reported_cards) or
            _cards_changed(current_upper, self._last_reported_upper) or
            _cards_changed(current_lower, self._last_reported_lower)):
            self._last_reported_cards = list(current_my)
            self._last_reported_upper = list(current_upper)
            self._last_reported_lower = list(current_lower)
        else:
            # No meaningful change — keep previous stable result
            result["my_cards"] = list(self._last_reported_cards) if self._last_reported_cards else []
            result["upper_player_last"] = list(self._last_reported_upper) if self._last_reported_upper else []
            result["lower_player_last"] = list(self._last_reported_lower) if self._last_reported_lower else []

        return result


def test_recognizer():
    """Test the recognizer."""
    print("=" * 50)
    print(" Testing Card Recognizer (Pure YOLO)")
    print("=" * 50)

    recognizer = CardRecognizer(
        model_path="models/yolov8_cards.pt",
        confidence_threshold=0.5
    )

    import os
    test_img_path = "test_doudizhu.png"
    if os.path.exists(test_img_path):
        import cv2
        img = _get_cv2().imread(test_img_path)
        if img is not None:
            detections = recognizer.detect_cards(img)
            print(f"\nDetected {len(detections)} cards:")
            for d in detections:
                print(f"  {d['card']}: conf={d['confidence']:.3f}, x={d['center_x']}")
    else:
        print(f"Test image not found: {test_img_path}")


if __name__ == "__main__":
    test_recognizer()
