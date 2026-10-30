"""Drawing of the counting line, tracked holes and counters on a frame."""
import cv2


def draw_overlay(frame, result, counter, line, hole_count, anode_count, fps):
    """
    Return an annotated copy of frame.

    line is (x1, y, x2). Only detections the counter accepts are drawn, so the
    picture shows what the counting logic actually used.
    """
    x1, y, x2 = line
    annotated = frame.copy()
    cv2.line(annotated, (x1, y), (x2, y), (0, 0, 255), 3)

    for det in result.detections:
        if not counter.accepts(det):
            continue
        cv2.rectangle(annotated, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), (255, 0, 0), 2)
        cv2.putText(annotated, f"ID:{det.track_id}", (int(det.x1), int(det.y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(annotated, f"Holes: {hole_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(annotated, f"Anodes: {anode_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return annotated
