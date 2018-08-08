package FaceDetector;

class HaarFeature {
  static final int H2D = 0;
  static final int V2D = 1;
  static final int H3D = 2;
  static final int V3D = 3;
  static final int X4D = 4;

  private int type;
  private int xPos;
  private int yPos;
  private int width;
  private int height;
  private int polarity;
  private int value;
  private int margin;
  private double error;
  private boolean[] clasification;

  HaarFeature(int type, int xPos, int yPos, int width, int height, int[] threshold) {
    this.type = type;
    this.xPos = xPos;
    this.yPos = yPos;
    this.width = width;
    this.height = height;
    this.value = threshold[0];
    this.polarity = threshold[1];
    this.margin = threshold[2];
  }

  public HaarFeature(HaarFeature feature, double scale) {
    this.type = feature.type;
    this.xPos = (int) (feature.xPos * scale);
    this.yPos = (int) (feature.yPos * scale);
    this.width = (int) (feature.width * scale);
    this.height = (int) (feature.height * scale);
    this.value = feature.value;
    this.margin = feature.margin;
    this.polarity = feature.polarity;
    this.error = feature.error;
  }

  public HaarFeature(int type, int x, int y, int w, int h, int value, int p, int margin) {
    this.type = type;
    this.xPos = x;
    this.yPos = y;
    this.width = w;
    this.height = h;
    this.value = value;
    this.polarity = p;
    this.margin = margin;
  }

  @Override
  public String toString() {
    return "Feature: Type: " + type + "   X: " + xPos + "   Y: " + yPos + "   W: " + width + "   H: " +
            height + "   Value: " + value + "   P: " + polarity + "   Error: " + error + "  Margin: " + margin;
  }

  int getValue() {
    return value;
  }

  int getType() {
    return type;
  }

  int getX() {
    return xPos;
  }

  int getY() {
    return yPos;
  }

  int getW() {
    return width;
  }

  int getH() {
    return height;
  }

  void setError(double error) {
    this.error = error;
  }

  int getPolarity() {
    return polarity;
  }

  public double getError() {
    return error;
  }

  public void setClasification(boolean[] clasification) {
    this.clasification = clasification;
  }

  public boolean[] getClasification() {
    return clasification;
  }

  public int getMargin() {
    return margin;
  }
}
