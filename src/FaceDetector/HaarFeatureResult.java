package FaceDetector;

class HaarFeatureResult {
  private int imageId;
  private int value;
  private int polarity;
  private boolean isPositive;

  HaarFeatureResult(int value, int imageId, boolean isPositive) {
    this.value = value;
    this.imageId = imageId;
    this.isPositive = isPositive;
  }

  public HaarFeatureResult() {

  }

  int getValue() {
    return value;
  }

  public boolean isPositive() {
    return isPositive;
  }

  public int getImageId() {
    return imageId;
  }

  int getPolarity() {
    return polarity;
  }

  public void setParams(int value, int polarity) {
    this.value = value;
    this.polarity = polarity;
  }
}
