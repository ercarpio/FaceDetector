package FaceDetector;

import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.AbstractMap;
import java.util.Calendar;

public class FaceDetectorClient {
  public static void main(String[] args) {
    SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
    System.out.println("Training started " + sdf.format(Calendar.getInstance().getTime()));
    long lStart = System.nanoTime();

    //OUTPUT A FEATURE IN A SAMPLE IMAGE
    /*drawFeature("../input/cswasFaces/1034.pgm", "../input/f1.pgm");*/


    //NORMALIZE ALL THE IMAGES IN A DIRECTORY
    /*normalizeImage("../input/testFaces");
    normalizeImage("../input/testBkg");*/


    //PROCESS A SET OF INPUT IMAGES NAMED A TO X
    /*for (char c = 'a'; c < 'i'; c++) {
      processImage("C:\\Users\\eccar\\Cortana\\Grad School\\Box Sync\\Fall16\\CS830\\Project\\input\\" + c + ".pgm",
              "C:\\Users\\eccar\\Cortana\\Grad School\\Box Sync\\Fall16\\CS830\\Project\\input\\test.pgm");
    }*/


    //EVALUATE THE PERFORMANCE OF THE CLASSIFIER IN A GIVEN SET
    evaluateHaarFeatureSet("../input/testFaces", true);
    evaluateHaarFeatureSet("../input/testBkg", false);
    evaluateHaarFeatureSet("../input/cswNSF", true);
    evaluateHaarFeatureSet("../input/cswNSB", false);


    long lEnd = System.nanoTime();
    System.out.printf("time: %5f", (double)(lEnd -lStart)/1000000000);
    System.out.println("\nTraining finished " + sdf.format(Calendar.getInstance().getTime()));

  }

  private static void evaluateHaarFeatureSet(String path, boolean isPositive) {
    FaceDetector fd = new FaceDetector(FaceDetector.DB_PATH);
    fd.computeIntegralImages(path, isPositive);
    fd.evaluateFeatures();
  }

  private static void processImage(String imagePath, String tempImagePath) {
    FaceDetector fd = new FaceDetector(FaceDetector.DB_PATH);
//    fd.convertJPGtoPGM(imagePath, tempImagePath);
    fd.normalizeImage(imagePath, tempImagePath);
    fd.processImage(tempImagePath);
  }

  private static void normalizeImage(String path) {
    FaceDetector fd = new FaceDetector(FaceDetector.DB_PATH);
    fd.normalizeImages(path);
  }

  private static void drawFeature(String inputFile, String outputFile) {
    FaceDetector fd = new FaceDetector(FaceDetector.DB_PATH);
    HaarFeature feature = new HaarFeature(3,10,0,45,10,11029,-1,10996);
    fd.drawHaarFeatures(inputFile, outputFile, feature);
  }
}
