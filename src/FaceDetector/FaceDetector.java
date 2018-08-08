package FaceDetector;

import java.io.*;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.*;

public class FaceDetector {
  private static final String PGM_ENCODING = "ISO-8859-1";
  static final String POSITIVE_INPUT_DIR = "../input/smallFaces";
  static final String NEGATIVE_INPUT_DIR = "../input/smallBackground";
  static final String POSITIVE_INPUT_DIR_CSW = "../input/cswasAllSmallFaces";
  static final String NEGATIVE_INPUT_DIR_CSW = "../input/cswasAllSmallBackground";
  static final String POSITIVE_INPUT_DIR_TOY = "../input/cswasSmallFaces";
  static final String NEGATIVE_INPUT_DIR_TOY = "../input/cswasSmallBackground";
  static final String POSITIVE_INPUT_DIR_NSF = "../input/cswNSF";
  static final String NEGATIVE_INPUT_DIR_NSB = "../input/cswNSB";

  static final String OUTPUT_IMAGE_ORIGINAL = "../input/cswasSmallFaces/001.pgm";
  static final String OUTPUT_IMAGE = "../output/featuresCSW.pgm";
  static final String DB_PATH = "../input/fd.data";

  private static final double SCALE_FACTOR = 1.25;
  private static final double SHIFT_FACTOR = 1.5;
  private static final int INITIAL_WINDOW_SIZE = 24;

  private static final int ADA_BOOST_ROUNDS = 200;
  private static final int BOX_COLOR = 255;
  private static final int COLORED_BOX_COLOR = 0;
  private static final double NORMALIZING_MEAN = 120;
  private static final double NORMALIZING_VARIANCE = 4000;

  //IMPORTANT TO MODIFY IF THIS IS NOT THE ACTUAL PATH TO THE PROGRAM
  //THIS IS ONLY USED DURING JPG TO PGM CONVERSION
  private static final String IRFAN_PATH = "C:\\Program Files (x86)\\IrfanView\\i_view32.exe";

  private final ExecutorService exec;
  private int imageWidth;
  private int imageHeight;
  private int numPositiveSamples;
  private int numNegativeSamples;
  private int hitCount = 0;
  private int totalCount = 0;
  private HashMap<Integer, FDImage> images;
  private ConcurrentHashMap<HaarFeature, Integer> preliminaryFeatures;
  private HashSet<HaarFeature> selectedFeatures;

  FaceDetector(String dbPath) {
    images = new HashMap<>();
    selectedFeatures = new HashSet<>();
    exec = Executors.newFixedThreadPool(4);
    loadDB(dbPath);
    FDImage.restartID();
  }

  private void loadDB(String dbPath) {
    try {
      Scanner db = new Scanner(new File(dbPath));
      while (db.hasNext()) {
        String line = db.nextLine();
        String[] featureValues = line.split(" ");
        int type = 0;
        int x = 0;
        int y = 0;
        int w = 0;
        int h = 0;
        int value = 0;
        int margin = 0;
        int p = 0;
        double error = 0;
        for (int i = 0; i < featureValues.length; i++) {
          switch (featureValues[i]) {
            case "Type:":
              type = Integer.parseInt(featureValues[i + 1]);
              break;
            case "X:":
              x = Integer.parseInt(featureValues[i + 1]);
              break;
            case "Y:":
              y = Integer.parseInt(featureValues[i + 1]);
              break;
            case "W:":
              w = Integer.parseInt(featureValues[i + 1]);
              break;
            case "H:":
              h = Integer.parseInt(featureValues[i + 1]);
              break;
            case "Value:":
              value = Integer.parseInt(featureValues[i + 1]);
              break;
            case "P:":
              p = Integer.parseInt(featureValues[i + 1]);
              break;
            case "Error:":
              error = Double.parseDouble(featureValues[i + 1]);
              break;
            case "Margin:":
              margin = Integer.parseInt(featureValues[i + 1]);
              break;
          }
        }
        HaarFeature feature = new HaarFeature(type, x, y, w, h, value, p, margin);
        feature.setError(error);
        selectedFeatures.add(feature);
      }
    } catch (FileNotFoundException e) {
      System.out.println("Error loading the DB.");
      e.printStackTrace();
    }
  }

  void computeIntegralImages(String inputDirectory, boolean isPositive) {
    File inputDir = new File(inputDirectory);
    File[] imageList = inputDir.listFiles();
    if (imageList != null) {
      if (isPositive)
        numPositiveSamples = imageList.length;
      else
        numNegativeSamples = imageList.length;
      for (File image : imageList) {
        int[][] integralImage = computeIntegralImage(image);
        FDImage fdImage = new FDImage(image.getName(), integralImage, isPositive);
        images.put(fdImage.getId(), fdImage);
      }
    }
  }

  private int[][] computeIntegralImage(File image) {
    int[][] integralImage = null;
    //int[][] img = null;
    try {
      FileInputStream stream = new FileInputStream(image);
      InputStreamReader streamReader = new InputStreamReader(stream, Charset.forName(PGM_ENCODING));
      BufferedReader reader = new BufferedReader(streamReader);
      reader.readLine(); //Magic number (P5)
      reader.readLine(); //Irfanview credits
      String[] dimensions = reader.readLine().split(" ");
      reader.readLine(); //Pixel maximum value (255)
      imageWidth = Integer.parseInt(dimensions[0]);
      imageHeight = Integer.parseInt(dimensions[1]);
      integralImage = new int[imageWidth][imageHeight];
      //img = new int[imageWidth][imageHeight];
      int colCounter = 0;
      int rowCounter = 0;
      int pixel;
      while (rowCounter < imageHeight) {
        pixel = reader.read();
        //img[colCounter][rowCounter] = pixel;
        if (colCounter == 0 && rowCounter == 0) {
          integralImage[colCounter][rowCounter] = pixel;
        } else if (colCounter == 0) {
          integralImage[colCounter][rowCounter] = integralImage[colCounter][rowCounter - 1] + pixel;
        } else if (rowCounter == 0) {
          integralImage[colCounter][rowCounter] = integralImage[colCounter - 1][rowCounter] + pixel;
        } else {
          integralImage[colCounter][rowCounter] = pixel + integralImage[colCounter - 1][rowCounter] +
                  integralImage[colCounter][rowCounter - 1] - integralImage[colCounter - 1][rowCounter - 1];
        }
        colCounter++;
        if (colCounter == imageWidth) {
          rowCounter++;
          colCounter = 0;
        }
      }
      reader.close();
      stream.close();
    } catch (IOException e) {
      System.out.println("Error with image " + image.getName());
      e.printStackTrace();
    }
    return integralImage;
  }

  private void computeHaarFeatures() {
    selectedFeatures = new HashSet<>();
    initializeWeights();
    for (int i = 0; i < ADA_BOOST_ROUNDS; i++) {
      System.out.println("Starting round " + i);
      preliminaryFeatures = new ConcurrentHashMap<>();
      normalizeWeights();
      System.out.println("\tComputing H2D filters...");
      Future<?> fH2D = exec.submit(() -> computeFeatures(HaarFeature.H2D, 2, 1));
      System.out.println("\tComputing V2D filters...");
      Future<?> fV2D = exec.submit(() -> computeFeatures(HaarFeature.V2D, 1, 2));
      System.out.println("\tComputing H3D filters...");
      Future<?> fH3D = exec.submit(() -> computeFeatures(HaarFeature.H3D, 3, 1));
      System.out.println("\tComputing V3D filters...");
      Future<?> fV3D = exec.submit(() -> computeFeatures(HaarFeature.V3D, 1, 3));
      System.out.println("\tComputing 4D filters...");
      Future<?> fX4D = exec.submit(() -> computeFeatures(HaarFeature.X4D, 2, 2));
      try {
        fH2D.get();
        System.out.println("\tH2D filters computed.");
        fV2D.get();
        System.out.println("\tV2D filters computed.");
        fH3D.get();
        System.out.println("\tH3D filters computed.");
        fV3D.get();
        System.out.println("\tV3D filters computed.");
        fX4D.get();
        System.out.println("\t4D filters computed.");
        System.out.println("\tSelecting best filter...");
        double minError = Double.MAX_VALUE;
        HaarFeature selectedFeature = null;
        for (Map.Entry<HaarFeature, Integer> entry : preliminaryFeatures.entrySet()) {
          HaarFeature feature = entry.getKey();
          double error = feature.getError();
          if (error < minError) {
            minError = error;
            selectedFeature = feature;
          }
        }
        assert selectedFeature != null;
        updateWeights(minError, selectedFeature);
        selectedFeatures.add(selectedFeature);
        System.out.println(selectedFeature.toString());
      } catch (InterruptedException e) {
        System.out.println("Timeout exceeded!");
        e.printStackTrace();
      } catch (ExecutionException e) {
        System.out.println("Computation error!");
        e.printStackTrace();
      }
    }
    evaluateFeatures();
    saveDB();
    exec.shutdown();
  }

  private void saveDB() {
    try (Writer writer = new BufferedWriter(new OutputStreamWriter(
            new FileOutputStream(DB_PATH), Charset.defaultCharset()))) {
      for (HaarFeature feature : selectedFeatures) {
        writer.write(feature.toString() + "\n");
      }
    } catch (IOException e) {
      System.out.println("Error creating DB.");
      e.printStackTrace();
    }
  }

  void evaluateFeatures() {
    int correctCount = 0;
    int falsePositives = 0;
    int falseNegatives = 0;
    for (int i = 0; i < images.size(); i++) {
      double sumImage = 0;
      double sumAlfa = 0;
      for (HaarFeature f : selectedFeatures) {
        double alfa = Math.log((1 - f.getError()) / f.getError()) / 2;
        int h = calculateH(images.get(i).getIntegral(), f, f.getPolarity(), f.getValue(), imageWidth);
        sumImage += alfa * h;
        sumAlfa += alfa;
      }
      if (sumImage >= sumAlfa / 2) {
        if (images.get(i).isPositive()) {
          correctCount++;
        } else {
          falsePositives++;
        }
      } else {
        if (!images.get(i).isPositive()) {
          correctCount++;
        } else {
          falseNegatives++;
        }
      }
    }
    System.out.println("CORRECT: " + correctCount + "  FN: " + falseNegatives + "  FP: " + falsePositives);
  }

  private int calculateH(int[][] image, HaarFeature feature, int polarity,
                         double thresholdValue, int windowSize) {
    int h = 0;
    int imageValue = computeValue(image, feature.getType(), feature.getX(),
            feature.getY(), feature.getW(), feature.getH(), windowSize);
    if (polarity == 1) {
      if (imageValue <= thresholdValue)
        h = 1;
    } else {
      if (imageValue >= thresholdValue)
        h = 1;
    }
    return h;
  }

  private int computeValue(int[][] image, int type, int x, int y, int w, int h, int windowSize) {
    int value = Integer.MIN_VALUE;
    int supportSize;
    x = Math.round(x * windowSize / 24);
    y = Math.round(y * windowSize / 24);
    h = Math.round(h * windowSize / 24);
    w = Math.round(w * windowSize / 24);
    switch (type) {
      case HaarFeature.H2D:
        supportSize = 2 * w * h;
        h = Math.min(h, windowSize);
        if (w * 2 % 2 != 0)
          w++;
        while (w * 2 + x > windowSize)
          w -= 2;
        value = sumSquare(image, x, y, w, h) - sumSquare(image, x + w, y, w, h);
        value = (value * supportSize) / (2 * w * h);
        break;
      case HaarFeature.V2D:
        supportSize = 2 * w * h;
        w = Math.min(w, windowSize);
        if (h * 2 % 2 != 0)
          h++;
        while (h * 2 + y > windowSize)
          h -= 2;
        value = sumSquare(image, x, y, w, h) - sumSquare(image, x, y + h, w, h);
        value = (value * supportSize) / (2 * w * h);
        break;
      case HaarFeature.H3D:
        supportSize = 3 * w * h;
        h = Math.min(h, windowSize);
        if (w * 3 % 3 != 0)
          w = w + 3 - w * 3 % 3;
        while (w * 3 + x > windowSize)
          w -= 3;
        value = sumSquare(image, x, y, w, h) - sumSquare(image, x + w, y, w, h) +
                sumSquare(image, x + 2 * w, y, w, h);
        value = (value * supportSize) / (3 * w * h);
        break;
      case HaarFeature.V3D:
        supportSize = 3 * w * h;
        w = Math.min(w, windowSize);
        if (h * 3 % 3 != 0)
          h = h + 3 - h * 3 % 3;
        while (h * 3 + y > windowSize)
          h -= 3;
        value = sumSquare(image, x, y, w, h) - sumSquare(image, x, y + h, w, h) +
                sumSquare(image, x, y + 2 * h, w, h);
        value = (value * supportSize) / (3 * w * h);
        break;
      case HaarFeature.X4D:
        supportSize = 4 * w * h;
        if (h * 2 % 2 != 0)
          h++;
        while (h * 2 + y > windowSize)
          h -= 2;
        if (w * 2 % 2 != 0)
          w++;
        while (w * 2 + x > windowSize)
          w -= 2;
        value = sumSquare(image, x, y, w, h) - sumSquare(image, x + w, y, w, h) -
                sumSquare(image, x, y + h, w, h) + sumSquare(image, x + w, y + h, w, h);
        value = (value * supportSize) / (4 * w * h);
        break;
    }
    return value;
  }

  private void initializeWeights() {
    double initialPWeight = 1.0 / (numPositiveSamples * 2);
    double initialNWeight = 1.0 / (numNegativeSamples * 2);
    for (int i = 0; i < images.size(); i++) {
      FDImage image = images.get(i);
      if (image.isPositive())
        image.setWeight(initialPWeight);
      else
        image.setWeight(initialNWeight);
    }
  }

  private void normalizeWeights() {
    double wSum = 0;
    for (int i = 0; i < images.size(); i++)
      wSum += images.get(i).getWeight();
    for (int i = 0; i < images.size(); i++)
      images.get(i).normalizeWeight(wSum);
  }

  private void updateWeights(double minError, HaarFeature feature) {
    double beta = minError / (1 - minError);
    for (int i = 0; i < images.size(); i++) {
      boolean prediction = false;
      if (calculateH(images.get(i).getIntegral(), feature, feature.getPolarity(),
              feature.getValue(), imageWidth) == 1)
        prediction = true;
      if (prediction == images.get(i).isPositive())
        images.get(i).updateWeight(Math.pow(beta, 1));
      else
        images.get(i).updateWeight(Math.pow(beta, 0));
    }
  }

  private void computeFeatures(int type, int wMultiplier, int hMultiplier) {
    for (int x = 0; x < imageWidth; x++) {
      for (int y = 0; y < imageHeight; y++) {
        for (int w = 1; w * wMultiplier + x <= imageWidth; w++) {
          for (int h = 1; h * hMultiplier + y <= imageHeight; h++) {
            double tp = 0;
            double tn = 0;
            PriorityQueue<HaarFeatureResult> sortedValues = new PriorityQueue<>(
                    (Comparator.comparingInt(o -> o.getValue())));
            for (int i = 0; i < images.size(); i++) {
              FDImage image = images.get(i);
              int value = computeValue(image.getIntegral(), type, x, y, w, h, imageWidth);
              //if (x == 2 && y == 5 && w == 6 && h == 3) System.out.println("validate " + value);
              HaarFeatureResult result = new HaarFeatureResult(value, i, image.isPositive());
              sortedValues.add(result);
              if (image.isPositive())
                tp += image.getWeight();
              else
                tn += image.getWeight();
            }
            int[] thresholdInfo = getThreshold(tp, tn, sortedValues);
            HaarFeature feature = new HaarFeature(type, x, y, w, h, thresholdInfo);
            double error = 0;
            int polarity = feature.getPolarity();
            int thresholdValue = feature.getValue();
            for (int j = 0; j < images.size(); j++) {
              FDImage image = images.get(j);
              int hVal = calculateH(image.getIntegral(), feature, polarity, thresholdValue, imageWidth);
              int yHat = 0;
              if (image.isPositive())
                yHat = 1;
              error += image.getWeight() * Math.abs(hVal - yHat);
            }
            feature.setError(error);
            preliminaryFeatures.put(feature, type);
          }
        }
      }
    }
  }

  private int[] getThreshold(
          double tp, double tn, PriorityQueue<HaarFeatureResult> sortedValues) {
    int polarity = 0;
    int value = 0;
    int margin = 0;
    int minValue = Integer.MAX_VALUE;
    int maxValue = Integer.MIN_VALUE;
    double sp = 0;
    double sn = 0;
    double minError = Integer.MAX_VALUE;
    HaarFeatureResult current;
    while (!sortedValues.isEmpty()) {
      current = sortedValues.poll();
      int currValue = current.getValue();
      if (current.isPositive()) {
        sp += images.get(current.getImageId()).getWeight();
        if (currValue < minValue)
          minValue = currValue;
        else if (currValue > maxValue)
          maxValue = currValue;
      } else {
        sn += images.get(current.getImageId()).getWeight();
      }
      double negative = sp + tn - sn;
      double positive = sn + tp - sp;
      double error = Math.min(positive, negative);
      if (error < minError) {
        minError = error;
        if (error == positive) {
          polarity = 1;
          margin = minValue;
        }
        else {
          polarity = -1;
          margin = maxValue;
        }
        value = current.getValue();
      }
    }
    return new int[]{value, polarity, margin};
  }

  private int sumSquare(int[][] integralImage, int x, int y, int w, int h) {
    int value;
    if (x != 0 && y != 0) {
      value = integralImage[x + w - 1][y + h - 1] - integralImage[x - 1][y + h - 1] -
              integralImage[x + w - 1][y - 1] + integralImage[x - 1][y - 1];
    } else if (y != 0) {
      value = integralImage[x + w - 1][y + h - 1] - integralImage[x + w - 1][y - 1];
    } else if (x != 0) {
      value = integralImage[x + w - 1][y + h - 1] - integralImage[x - 1][y + h - 1];
    } else {
      value = integralImage[x + w - 1][y + h - 1];
    }
    return value;
  }

  private void drawHaarFeatures(String inputFile, String outputFile) {
    int[][] pixels = null;
    String l1 = "";
    String l2 = "";
    String l3 = "";
    String l4 = "";
    try {
      InputStream stream = new FileInputStream(new File(inputFile));
      BufferedReader reader = new BufferedReader(new InputStreamReader(stream, Charset.forName(PGM_ENCODING)));
      l1 = reader.readLine();
      l2 = reader.readLine();
      l3 = reader.readLine();
      l4 = reader.readLine();
      String[] dimensions = l3.split(" ");
      int b;
      int counter = 0;
      int row = 0;
      int numCol = Integer.parseInt(dimensions[0]);
      int numRow = Integer.parseInt(dimensions[1]);
      pixels = new int[numCol][numRow];
      while (row < numRow) {
        b = reader.read();
        pixels[counter][row] = b;
        counter++;
        if (counter == numCol) {
          counter = 0;
          row++;
        }
      }
      reader.close();
      stream.close();
    } catch (FileNotFoundException e) {
      System.out.println("Image not found: " + inputFile);
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
    for (HaarFeature feature : selectedFeatures) {
      drawFeature(feature, pixels);
    }
    try {
      OutputStream out = new FileOutputStream(outputFile);
      BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, Charset.forName(PGM_ENCODING)));
      writer.write(l1 + "\n");
      writer.write(l2 + "\n");
      writer.write(l3 + "\n");
      writer.write(l4 + "\n");
      assert pixels != null;
      for (int r = 0; r < pixels[0].length; r++) {
        for (int c = 0; c < pixels.length; c++) {
          writer.write(pixels[c][r]);
        }
      }
      writer.close();
      out.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private void drawFeature(HaarFeature feature, int[][] pixels) {
    int x = feature.getX();
    int y = feature.getY();
    int w = feature.getW();
    int h = feature.getH();
    switch (feature.getType()) {
      case HaarFeature.H2D:
        drawColoredBox(pixels, x + w, y, w, h, COLORED_BOX_COLOR);
        drawEmptyBox(pixels, x, y, w, h, true, true, false, true);
        break;
      case HaarFeature.V2D:
        drawColoredBox(pixels, x, y + h, w, h, COLORED_BOX_COLOR);
        drawEmptyBox(pixels, x, y, w, h, true, false, true, true);
        break;
      case HaarFeature.H3D:
        drawColoredBox(pixels, x + w, y, w, h, COLORED_BOX_COLOR);
        drawEmptyBox(pixels, x, y, w, h, true, true, false, true);
        drawEmptyBox(pixels, x + 2 * w, y, w, h, true, true, true, false);
        break;
      case HaarFeature.V3D:
        drawColoredBox(pixels, x, y + h, w, h, COLORED_BOX_COLOR);
        drawEmptyBox(pixels, x, y, w, h, true, false, true, true);
        drawEmptyBox(pixels, x, y + 2 * h, w, h, false, true, true, true);
        break;
      case HaarFeature.X4D:
        drawColoredBox(pixels, x + w, y, w, h, COLORED_BOX_COLOR);
        drawColoredBox(pixels, x, y + h, w, h, COLORED_BOX_COLOR);
        drawEmptyBox(pixels, x, y, w, h, true, false, false, true);
        drawEmptyBox(pixels, x + w, y + h, w, h, false, true, true, false);
        break;
    }
  }

  private void drawColoredBox(int[][] pixels, int x, int y, int w, int h, int coloredBoxColor) {
    for (int i = y; i <= y + h - 1; i++) {
      for (int j = x; j <= x + w - 1; j++) {
        pixels[j][i] = coloredBoxColor;
      }
    }
  }

  private void drawEmptyBox(int[][] pixels, int x, int y, int w, int h,
                            boolean N, boolean S, boolean E, boolean W) {
    drawColoredBox(pixels, x, y, w, h, BOX_COLOR);
    drawBoxMargins(pixels, x, y, w, h, N, S, E, W);
  }

  private void drawBoxMargins(int[][] pixels, int x, int y, int w, int h,
                              boolean N, boolean S, boolean E, boolean W) {
    for (int i = y; i <= y + h - 1; i++) {
      if (W)
        pixels[x][i] = COLORED_BOX_COLOR;
      if (E)
        pixels[x + w - 1][i] = COLORED_BOX_COLOR;
    }
    for (int i = x; i <= x + w - 1; i++) {
      if (N)
        pixels[i][y] = COLORED_BOX_COLOR;
      if (S)
        pixels[i][y + h - 1] = COLORED_BOX_COLOR;
    }
  }

  void drawHaarFeatures(String inputFile, String outputFile, HaarFeature feature) {
    selectedFeatures = new HashSet<>();
    selectedFeatures.add(feature);
    drawHaarFeatures(inputFile, outputFile);
  }

  private void computeIntegralImage(String file, String integralFile) {
    int width = 0;
    int height = 0;
    int[][] pixelValues = null;
    try {
      InputStream stream = new FileInputStream(file);
      BufferedReader reader = new BufferedReader(new InputStreamReader(stream, Charset.forName(PGM_ENCODING)));
      reader.readLine();
      reader.readLine();
      String[] dimensions = reader.readLine().split(" ");
      reader.readLine();
      int b;
      int colCounter = 0;
      int rowCounter = 0;
      width = Integer.parseInt(dimensions[0]);
      height = Integer.parseInt(dimensions[1]);
      pixelValues = new int[width][height];
      while (rowCounter < height) {
        b = reader.read();
        pixelValues[colCounter][rowCounter] = b;
        colCounter++;
        if (colCounter == width) {
          rowCounter++;
          colCounter = 0;
        }
      }
      reader.close();
      stream.close();
    } catch (FileNotFoundException e) {
      System.out.println("Image not found: " + file);
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }

    try {
      int colCounter = 1;
      int rowCounter = 0;
      OutputStream out = new FileOutputStream(integralFile);
      BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, Charset.forName(PGM_ENCODING)));
      while (rowCounter < height) {
        if (colCounter == 0) {
          pixelValues[colCounter][rowCounter] += pixelValues[colCounter][rowCounter - 1];
        } else if (rowCounter == 0) {
          pixelValues[colCounter][rowCounter] += pixelValues[colCounter - 1][rowCounter];
        } else {
          pixelValues[colCounter][rowCounter] += pixelValues[colCounter - 1][rowCounter] +
                  pixelValues[colCounter][rowCounter - 1] - pixelValues[colCounter - 1][rowCounter - 1];
        }
        writer.write(pixelValues[colCounter][rowCounter] + " ");
        colCounter++;
        if (colCounter == width) {
          colCounter = 0;
          rowCounter++;
          writer.write("\n");
        }
      }
      writer.close();
      out.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  void computeAverageImage(String inputPath, String outputPath) {
    String l1 = "";
    String l2 = "";
    String l3 = "";
    String l4 = "";
    File dir = new File(inputPath);
    File[] images = dir.listFiles();
    if (images != null) {
      ArrayList<Double> averagePixelValue = null;
      for (File f : images) {
        try {
          InputStream stream = new FileInputStream(f);
          BufferedReader reader = new BufferedReader(new InputStreamReader(stream, Charset.forName(PGM_ENCODING)));
          l1 = reader.readLine();
          l2 = reader.readLine();
          l3 = reader.readLine();
          l4 = reader.readLine();
          String[] dimensions = l3.split(" ");
          int b;
          int counter = 0;
          int totalPixels = Integer.parseInt(dimensions[0]) * Integer.parseInt(dimensions[1]);
          if (averagePixelValue == null) {
            averagePixelValue = new ArrayList<>();
            for (int i = 0; i < totalPixels; i++)
              averagePixelValue.add(0.0);
          }
          while (counter < totalPixels) {
            b = reader.read();
            averagePixelValue.set(counter, averagePixelValue.get(counter) + (double) (b));
            counter++;
          }
          reader.close();
          stream.close();
        } catch (FileNotFoundException e) {
          System.out.println("Image not found: " + f.getName());
          e.printStackTrace();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
      try {
        OutputStream out = new FileOutputStream(outputPath);
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, Charset.forName(PGM_ENCODING)));
        writer.write(l1 + "\n");
        writer.write(l2 + "\n");
        writer.write(l3 + "\n");
        writer.write(l4 + "\n");
        if (averagePixelValue != null) {
          for (Double d : averagePixelValue) {
            writer.write((int) (d / images.length));
          }
        }
        writer.close();
        out.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  void printImageStatistics(String path) {
    File dir = new File(path);
    File[] images = dir.listFiles();
    if (images != null) {
      int numImages = images.length;
      int minWidth = Integer.MAX_VALUE;
      int minHeight = Integer.MAX_VALUE;
      int maxWidth = Integer.MIN_VALUE;
      int maxHeight = Integer.MIN_VALUE;
      int avgWidth = 0;
      int avgHeight = 0;
      int width, height;
      for (File f : images) {
        try {
          Scanner fileScan = new Scanner(new FileReader(f));
          fileScan.nextLine();
          //fileScan.nextLine();
          width = fileScan.nextInt();
          height = fileScan.nextInt();
          if (width > maxWidth)
            maxWidth = width;
          if (height > maxHeight)
            maxHeight = height;
          if (width < minWidth)
            minWidth = width;
          if (height < minHeight)
            minHeight = height;
          avgWidth += width;
          avgHeight += height;
          fileScan.close();
        } catch (FileNotFoundException e) {
          System.out.println("Image not found: " + f.getName());
          e.printStackTrace();
        }
      }
      avgWidth = avgWidth / numImages;
      avgHeight = avgHeight / numImages;
      System.out.println("Number of images: " + numImages);
      System.out.println("Avg W: " + avgWidth + "\tAvg H: " + avgHeight);
      System.out.println("Min W: " + minWidth + "\tAvg H: " + minHeight);
      System.out.println("Max W: " + maxWidth + "\tAvg H: " + maxHeight);
    }
  }

  void normalizeImages(String inputDirectory) {
    File inputDir = new File(inputDirectory);
    File[] imageList = inputDir.listFiles();
    if (imageList != null) {
      for (File image : imageList) {
        normalizeImage(image.getAbsolutePath(), image.getAbsolutePath());
      }
    }
  }

  void normalizeImage(String inputFile, String outputFile) {
    int[][] img = null;
    try {
      File image = new File(inputFile);
      FileInputStream stream = new FileInputStream(image);
      InputStreamReader streamReader = new InputStreamReader(stream, Charset.forName(PGM_ENCODING));
      BufferedReader reader = new BufferedReader(streamReader);
      reader.readLine(); //Magic number (P5)
      reader.readLine(); //Irfanview credits
      String[] dimensions = reader.readLine().split(" ");
      reader.readLine(); //Pixel maximum value (255)
      imageWidth = Integer.parseInt(dimensions[0]);
      imageHeight = Integer.parseInt(dimensions[1]);
      img = new int[imageWidth][imageHeight];
      int pixel;
      double mean = 0;
      for (int rowCounter = 0; rowCounter < imageHeight; rowCounter++) {
        for (int colCounter = 0; colCounter < imageWidth; colCounter++) {
          pixel = reader.read();
          img[colCounter][rowCounter] = pixel;
          mean += pixel;
        }
      }
      mean = mean / (imageHeight * imageWidth);
      double variance = calculateVariance(img, mean);
      for (int rowCounter = 0; rowCounter < imageHeight; rowCounter++) {
        for (int colCounter = 0; colCounter < imageWidth; colCounter++) {
          pixel = img[colCounter][rowCounter];
          double common = Math.sqrt(NORMALIZING_VARIANCE * Math.pow(pixel - mean, 2) / variance);
          int n = 0;
          if (pixel > mean) {
            n = (int) (NORMALIZING_MEAN + common);
          } else {
            n = (int) (NORMALIZING_MEAN - common);
          }
          if (n > 255)
            n = 255;
          else if (n < 0)
            n = 0;
          img[colCounter][rowCounter] = n;
        }
      }
      reader.close();
      stream.close();
    } catch (IOException e) {
      System.out.println("Error with image " + inputFile);
      e.printStackTrace();
    }
    try {
      OutputStream out = new FileOutputStream(outputFile);
      BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, Charset.forName(PGM_ENCODING)));
      writer.write("P5\n");
      writer.write("#\n");
      writer.write(imageWidth + " " + imageHeight + "\n");
      writer.write("255\n");
      assert img != null;
      for (int r = 0; r < img[0].length; r++) {
        for (int c = 0; c < img.length; c++) {
          writer.write(img[c][r]);
        }
      }
      writer.close();
      out.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private double calculateVariance(int[][] img, double mean) {
    double variance = 0;
    for (int rowCounter = 0; rowCounter < imageHeight; rowCounter++) {
      for (int colCounter = 0; colCounter < imageWidth; colCounter++) {
        variance += Math.pow(img[colCounter][rowCounter] - mean, 2);
      }
    }
    return variance / (imageWidth * imageHeight - 1);
  }

  public static void main(String[] args) {
    SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
    System.out.println("Training started " + sdf.format(Calendar.getInstance().getTime()));
    FaceDetector detector = new FaceDetector(FaceDetector.DB_PATH);
    System.out.println("Computing integral images... ");
    detector.computeIntegralImages(POSITIVE_INPUT_DIR_NSF, true);
    detector.computeIntegralImages(NEGATIVE_INPUT_DIR_NSB, false);
    System.out.println("Computing Haar filters... ");
    detector.computeHaarFeatures();
    System.out.println("Haar features computed.");
    detector.drawHaarFeatures(OUTPUT_IMAGE_ORIGINAL, OUTPUT_IMAGE);
    System.out.println("Training completed " + sdf.format(Calendar.getInstance().getTime()));
  }

  void convertJPGtoPGM(String imagePath, String tempImagePath) {
    try {
      Runtime run = Runtime.getRuntime();
      Process pr = run.exec(IRFAN_PATH + " " + imagePath + " /convert=" + tempImagePath);
      pr.waitFor();
    } catch (IOException | InterruptedException e) {
      System.out.println("Error converting image from JPG to PGM.");
      e.printStackTrace();
      System.exit(1);
    }
  }

  private void evaluateFeatures(int[][] image, int[][] integralImage, FDWindow window,
                                HashSet<HaarFeature> scaledFeatures,
                                PriorityQueue<FDWindow> selectedWindows) {
    double sumImage = 0;
    double sumAlfa = 0;
    for (HaarFeature f : scaledFeatures) {
      double alfa = Math.log((1 - f.getError()) / f.getError()) / 2;
      int h = calculateH(integralImage, f, f.getPolarity(), f.getValue(), window.w);
      sumImage += alfa * h;
      sumAlfa += alfa;
    }
    if (sumImage >= sumAlfa / 2) {
      hitCount++;
      selectedWindows.add(window);
    }
    else
      totalCount++;
  }

  private void saveImage(int[][] image, String outputFile) {
    try {
      OutputStream out = new FileOutputStream(outputFile);
      BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, Charset.forName(PGM_ENCODING)));
      writer.write("P5\n");
      writer.write("#\n");
      writer.write(imageWidth + " " + imageHeight + "\n");
      writer.write("255\n");
      assert image != null;
      for (int r = 0; r < image[0].length; r++) {
        for (int c = 0; c < image.length; c++) {
          writer.write(image[c][r]);
        }
      }
      writer.close();
      out.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private int[][] openImage(String inputFile) {
    int[][] img = null;
    try {
      File image = new File(inputFile);
      FileInputStream stream = new FileInputStream(image);
      InputStreamReader streamReader = new InputStreamReader(stream, Charset.forName(PGM_ENCODING));
      BufferedReader reader = new BufferedReader(streamReader);
      reader.readLine(); //Magic number (P5)
      reader.readLine(); //Irfanview credits
      String[] dimensions = reader.readLine().split(" ");
      reader.readLine(); //Pixel maximum value (255)
      imageWidth = Integer.parseInt(dimensions[0]);
      imageHeight = Integer.parseInt(dimensions[1]);
      img = new int[imageWidth][imageHeight];
      int pixel;
      for (int rowCounter = 0; rowCounter < imageHeight; rowCounter++) {
        for (int colCounter = 0; colCounter < imageWidth; colCounter++) {
          pixel = reader.read();
          img[colCounter][rowCounter] = pixel;
        }
      }
      reader.close();
      stream.close();
    } catch (IOException e) {
      System.out.println("Error with image " + inputFile);
      e.printStackTrace();
    }
    return img;
  }

  void processImage(String imagePath) {
    PriorityQueue<FDWindow> selectedWindows = new PriorityQueue<FDWindow>(
            (o1, o2) -> {
              if (o2.getArea() != o1.getArea())
                return o2.getArea() - o1.getArea();
              else if (o1.x != o2.x)
                return o1.x - o2.x;
              else
                return o1.y - o2.y;
            });
    int[][] image = openImage(imagePath);
    int[][] integralImage = computeIntegralImage(new File(imagePath));
    for (double x = 0; (int) x < integralImage.length; x = x + SHIFT_FACTOR) {
      for (double y = 0; (int) y < integralImage[0].length; y = y + SHIFT_FACTOR) {
        double h = INITIAL_WINDOW_SIZE;
        for (double w = INITIAL_WINDOW_SIZE; ((int) (w + x)-1 < integralImage.length) &&
                ((int) (y + h)-1 < integralImage[0].length); w = w * SCALE_FACTOR, h = w) {
          evaluateFeatures(image, integralImage, new FDWindow((int) x, (int) y,
                  (int) w, (int) h, 0), selectedFeatures, selectedWindows);
        }
      }
    }
    System.out.println(hitCount);
    System.out.println(totalCount);
    PriorityQueue<FDWindow> finalWindows = postProcessWindows(selectedWindows);
    drawWindows(finalWindows, image);
    saveImage(image, "../input/" + System.currentTimeMillis() + ".pgm");
  }

  private void drawWindows(PriorityQueue<FDWindow> finalWindows, int[][] image) {
    while(!finalWindows.isEmpty()) {
      FDWindow window = finalWindows.poll();
      drawBoxMargins(image, window.x, window.y, window.w, window.h, true, true, true, true);
    }
  }

  private PriorityQueue<FDWindow> postProcessWindows(PriorityQueue<FDWindow> selectedWindows) {
    int areaThreshold = 0;
    int distanceThreshold = 5;
    PriorityQueue<FDWindow> finalWindows = new PriorityQueue<FDWindow>(
            ((o1, o2) -> o2.confidence - o1.confidence));
    while (!selectedWindows.isEmpty()) {
      int confidence = 0;
      FDWindow current = selectedWindows.poll();
      while (! selectedWindows.isEmpty() &&
              current.getArea() - selectedWindows.peek().getArea() <= areaThreshold &&
              current.getPoint().distance(selectedWindows.peek().getPoint()) <= distanceThreshold) {
        selectedWindows.poll();
        confidence++;
      }
      if (confidence > 2) {
        boolean toAdd = true;
        FDWindow toDelete = null;
        FDWindow newW = null;
        current.setConfidence(confidence);
        for (FDWindow w: finalWindows) {
          if (w.overlaps(current)) {
            newW = w.contains(current);
            if (newW == null) {
            } else if (current == newW){
              toDelete = w;
              newW.merge(w);
            } else {
              toDelete = w;
              newW.merge(current);
            }
          }
        }
        if (newW != null)
          finalWindows.add(newW);
        else
          finalWindows.add(current);
        if (toDelete != null)
          finalWindows.remove(toDelete);
      }
    }
    boolean mergePerformed = true;
    while (mergePerformed) {
      mergePerformed = false;
      ArrayList<FDWindow> fWindows = new ArrayList<FDWindow>(finalWindows);
      for (int i = 0; i < fWindows.size() - 1; i++) {
        for (int j = i + 1; j < fWindows.size(); j++) {
          if ((!mergePerformed) && (fWindows.get(i).contains(fWindows.get(j)) != null)) {
            FDWindow newWindow = fWindows.get(i).merge(fWindows.get(j));
            finalWindows.remove(fWindows.get(i));
            finalWindows.remove(fWindows.get(j));
            finalWindows.add(newWindow);
            mergePerformed = true;
          }
        }
      }
    }
    return finalWindows;
  }
}