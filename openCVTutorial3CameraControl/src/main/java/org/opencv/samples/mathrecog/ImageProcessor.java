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
        erode(patient, patient, getStructuringElement(MORPH_RECT, new Size(2, 4)));
        imwrite("/sdcard/MathRecogData/__ats__" + debug, patient);

        // find lines
        Mat zeros = new Mat(patient.rows(), patient.cols(), patient.type(), new Scalar(255, 255, 255));
        subtract(zeros, patient, patient);
        for(int t = 0; t < 2; t++) {
            Mat lines = new Mat();
            HoughLinesP(patient, lines, 1, Math.PI / 180, 50, patient.height() / 2, 7);
            Log.i("Lines count:", String.valueOf(lines.cols()));
            if (lines.cols() > 10) {
                for (int i = 0; i < lines.cols(); i++) {
                    double[] val = lines.get(0, i);
                    line(patient, new Point(val[0], val[1]), new Point(val[2], val[3]), new Scalar(0, 0, 0), 3);
                    Log.i("New Line:", "("+val[0]+","+val[1]+") - ("+val[2]+","+val[3]+")");
                }
            }
        }
        // connect closest black points
        int w = patient.cols();
        int h = patient.rows();
        for (int j = 0; j < h - (h % 9); j += 9 ) {
            for (int k = 0; k < w - (w % 9); k+=9 ) {
                Mat tmp = patient.submat(j, j+9, k, k+9);
                double[] sum = getSum(tmp);
                if (sum > 255 * 6) {
                    rectangle(patient, new Point(k, j), new Point(k + 8, j + 8), new Scalar(255, 255, 255), -1);
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
        for (int i = 0; i < conts.size(); i++) {
            aver += arcLength(new MatOfPoint2f(conts.get(i).toArray()), false);
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
        Size s1 = new Size();
        Point p10 = new Point();
        separatedConts.get(20).locateROI(s1, p10);
        Size s2 = new Size();
        Point p20 = new Point();
        separatedConts.get(21).locateROI(s2, p20);
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
        double[] out = new double[10]; // 1 - 1,1, 2 - 1,2, 3 - 1,3, 4 - 2,1.....
        double sum = 0;
        for(int i = 0; i < m.cols(); i++){
            for(int t = 0; t < m.rows(); t++){
                double[] data = m.get(t, i);
                sum += data[0]; //for an image in grayscale
            }
        }
        return sum;
    }

    private int getMaxSum(Mat m){
        return m.cols()*m.rows()*255; //grayscale
    }
}
