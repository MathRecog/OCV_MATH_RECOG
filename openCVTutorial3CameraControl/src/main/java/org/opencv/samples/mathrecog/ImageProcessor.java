package org.opencv.samples.mathrecog;

import android.os.Environment;
import android.util.Log;

import org.opencv.core.*;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

import static org.opencv.core.Core.*;
import static org.opencv.highgui.Highgui.*;
import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.photo.Photo.*;

/**
 * Created by VladimirGerasimov on 05.02.2015.
 */
public class ImageProcessor {
    private Mat photo;
    String debug;

    ImageProcessor(String img) {
        photo = imread(img, CV_LOAD_IMAGE_COLOR);
        if (photo.empty()) {
            throw new Error("Cannot find the image file: " + img);
        }
        File f = new File(img);
        debug = f.getName();
    }

    public void Process() {
        Mat patient = new Mat(photo.rows(), photo.cols(), photo.type());
        photo.copyTo(patient);
        // 1st PART: image processing, finding symbols on the image and crop them

        // convert to grayscale, make it contrast
        cvtColor(patient, patient, COLOR_BGR2GRAY, 0);
        patient.convertTo(patient, -1, 2.05, -0.5);

        Mat symbolsCont = new Mat(photo.rows(), photo.cols(), photo.type());
        patient.copyTo(symbolsCont); // we are going to extract symbols from this matrix

        // thersh hold (make the image binary)
        adaptiveThreshold(patient, patient, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 35, 35);
//        erode(patient, patient, getStructuringElement(MORPH_RECT, new Size(2, 4)));

        imwrite("/sdcard/MathRecogData/__ats__" + debug, patient);

        // find lines
        Mat zeros = new Mat(patient.rows(), patient.cols(), patient.type(), new Scalar(255, 255, 255));
        subtract(zeros, patient, patient);
        for(int t = 0; t < 3; t++) {
            Mat lines = new Mat();
            HoughLinesP(patient, lines, 5, Math.PI / 180, 30, 200, 15);
            Log.i("Lines found", String.valueOf(lines.cols()));
            if (lines.cols() > 17) {
                for (int i = 0; i < lines.cols(); i++) {
                    double[] val = lines.get(0, i);
                    // y = kx + b
                    double k = (val[1] - val[3]) / (val[0] - val[2]);
                    double b = val[1] - k*val[0];
                    line(photo, new Point(0, (int)b), new Point(photo.cols(), (int)(k*photo.cols() + b) ), new Scalar(0, 200, 100), 4);
                    line(patient, new Point(0, (int)b), new Point(photo.cols(), (int)(k*photo.cols() + b) ), new Scalar(0, 0, 0), 4);
                }
            }
        }
        imwrite("/sdcard/MathRecogData/__lines__" + debug, photo);

        // connect closest black points
        for(int iter = 0; iter < 2; iter++ ) {
            int w = patient.cols() + iter * 5;
            int h = patient.rows() + iter * 5;
            int index = (3+iter);
            int koef = index*index;
            for (int j = 0; j < h - (h % koef); j += koef) {
                for (int k = 0; k < w - (w % koef); k += koef) {
                    Mat tmp = patient.submat(j, j + koef, k, k + koef);
                    double[] sum = getSum(tmp, koef);
                    if (sum[0] > 255 * koef * koef * 0.15) {
                        for (int x = 1; x < koef; x++) {
                            if (sum[x] >= 4 * 255) {
                                int x1 = k + ((x - 1) % 3) * 3 + 1;
                                int y1 = j + ((int) ((x - 1) / 3)) * 3 + 1;
                                for (int y = x + 1; y < 10; y++) {
                                    if (sum[y] >= 4 * 255) {
                                        int x2 = k + ((y - 1) % 3) * 3 + 1;
                                        int y2 = j + ((int) ((y - 1) / 3)) * 3 + 1;
                                        line(patient, new Point(x1, y1), new Point(x2, y2), new Scalar(255, 255, 255), 2);
                                    }
                                }
                            }
                        }
                    } else {
                        rectangle(patient, new Point(k, j), new Point(k + koef-1, j + koef-1), new Scalar(0, 0, 0), -1);
                    }
                }
            }
        }

        imwrite("/sdcard/MathRecogData/__algo__" + debug, patient);


        // find contours
        Mat hierarchy = new Mat();
        List<MatOfPoint> conts = new ArrayList<MatOfPoint>();
        findContours(patient, conts, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS);

        // get average contour length
        double aver = 0.0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < conts.size(); i++) {
            double value = arcLength(new MatOfPoint2f(conts.get(i).toArray()), false);
            aver += arcLength(new MatOfPoint2f(conts.get(i).toArray()), false);
            if( value > max ) { max = value; }
            if( value < min ) { min = value; }
        }
        aver = aver / conts.size();
        // get submats by contours bounding rectangles
        List<Mat> separatedConts = new ArrayList<Mat>();
        List<MatOfPoint> contsLong = new ArrayList<MatOfPoint>();
        for (int i = 0; i < conts.size(); i++) {
            if (arcLength(new MatOfPoint2f(conts.get(i).toArray()), false) > aver * 0.3) {
                contsLong.add(conts.get(i));
                Rect rect = boundingRect(conts.get(i));
                rectangle(patient, rect.tl(), rect.br(), new Scalar(255, 255, 255));
                Mat cropped = symbolsCont.submat(rect);
                separatedConts.add(cropped);
            }
        }

        imwrite("/sdcard/MathRecogData/__rect__" + debug, patient);
        //       Mat test = symbolsCont.submat();
        // 2nd PART: try to recognize symbols

        List<String> chars = new ArrayList<String>();

        for (int i = 0; i < separatedConts.size(); i++) {
            Neuro n = new Neuro(separatedConts.get(i));
            String ch = n.recognize();
            if (!ch.equals("FAIL")) {
                chars.add(i, ch);
            } else {

                // merge two contours into one, check once again
            }
        }


        // 3rd PART: generate a FORMULA

    }

    private double[] getSum(Mat m) {
        double[] out = new double[10]; // 0 - 1,1, 1 - 1,2, 2 - 1,3, 3 - 2,1.....
        for(int k = 0; k < 10; k++) { out[k] = 0; }

        for(int i = 0; i < m.cols(); i++){
            for(int t = 0; t < m.rows(); t++){
                double[] data = m.get(t, i);
                out[0] += data[0]; //for an image in grayscale
                int index = (int)(i / 3) + 3 * (int)(t / 3);
                out[index] += data[0];
            }
        }
        return out;
    }

    private int getMaxSum(Mat m){
        return m.cols()*m.rows()*255; //grayscale
    }
}
