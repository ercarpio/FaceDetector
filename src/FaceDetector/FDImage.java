package FaceDetector;

class FDImage {
  private static int ID;

  private final int id;
  private final String name;
  private final int[][] integral;
  private final boolean isPositive;
  private double weight;

  FDImage (String name, int[][] integral, boolean isPositive) {
    this.id = ID;
    this.name = name;
    this.isPositive = isPositive;
    this.integral = integral;
    ID++;
  }

  synchronized void setWeight(double w) {
    weight = w;
  }

  int getId() {
    return id;
  }

  String getName() {
    return name;
  }

  int[][] getIntegral() {
    return integral;
  }

  boolean isPositive() {
    return isPositive;
  }

  synchronized double getWeight() {
    return weight;
  }

  @Override
  public boolean equals(Object o) {
    if (o.getClass() != this.getClass())
      return false;
    FDImage i = (FDImage) o;
    return i.getId() == this.getId();
  }

  @Override
  public int hashCode() {
    return id;
  }

  synchronized void updateWeight(double beta) {
    weight = weight * beta;
  }

  synchronized void normalizeWeight(double wSum) {
    weight = weight / wSum;
  }

  public static void restartID() {
    ID = 0;
  }
}
