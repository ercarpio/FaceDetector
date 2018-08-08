package FaceDetector;

import java.awt.*;

public class FDWindow {
  int x;
  int y;
  int w;
  int h;
  int confidence;

  public FDWindow(int x, int y, int w, int h, int confidence) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.confidence = confidence;
  }

  @Override
  public String toString() {
    return x + " " + y + " " + w + " " + h;
  }

  @Override
  public boolean equals(Object o) {
    if (o.getClass() != this.getClass())
      return false;
    FDWindow w = (FDWindow) o;
    return (w.x == this.x) && (w.y == this.y) && (w.w == this.w) && (w.h == this.h);
  }

  @Override
  public int hashCode() {
    return x * y * w * h;
  }

  public int getArea() {
    return w * h;
  }

  public FDWindow contains(FDWindow w) {
    if ((this.x <= w.x + w.w / 2) && (this.x + this.w >= w.x + w.w / 2) &&
            (this.y <= w.y + w.h / 2) && (this.y + this.h >= w.y + w.h / 2)) {
      if (this.confidence < w.confidence)
        return w;
      else
        return this;
    }
    return null;
  }

  public int distanceToOrigin() {
    return (int) Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2));
  }

  public void setConfidence(int confidence) {
    this.confidence = confidence;
  }

  public boolean overlaps(FDWindow w) {
    int counter = 0;
    if ((this.x <= w.x) && (this.x + this.w >= w.x) &&
            (this.y <= w.y) && (this.y + this.h >= w.y))
      counter++;
    if ((this.x <= w.x + w.w) && (this.x + this.w >= w.x + w.w) &&
            (this.y <= w.y) && (this.y + this.h >= w.y))
      counter++;
    if ((this.x <= w.x) && (this.x + this.w >= w.x) &&
            (this.y <= w.y + w.h) && (this.y + this.h >= w.y + w.h))
      counter++;
    if ((this.x <= w.x + w.w) && (this.x + this.w >= w.x + w.w) &&
            (this.y <= w.y + w.h) && (this.y + this.h >= w.y + w.h))
      counter++;
      return counter > 0;
  }

  public Point getPoint() {
    return new Point(x, y);
  }

  public FDWindow merge(FDWindow w) {
      return new FDWindow((x + w.x) / 2, (y + w.y) / 2, (this.w + w.w) / 2,
              (h + w.h) / 2, Math.max(confidence, w.confidence));
  }
}
