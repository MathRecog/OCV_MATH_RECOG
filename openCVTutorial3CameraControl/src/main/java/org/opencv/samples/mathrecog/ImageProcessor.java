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
        Mat patient = photo;
        // 1st PART: image processing, finding symbols on the image and cut them

        // convert to grayscale, make it contrast
        cvtColor(patient, patient, COLOR_BGR2GRAY, 0);
        patient.convertTo(patient, -1, 1.95, -0.2);
        imwrite("/sdcard/MathRecogData/__ct__" + debug, patient);

        // thersh hold (make the image binary)
        adaptiveThreshold(patient, patient, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 15);
        erode(patient, patient, getStructuringElement(MORPH_RECT, new Size(2, 4)));
        imwrite("/sdcard/MathRecogData/__ats__" + debug, patient);

        // connect closest black points
        int w = patient.cols();
        int h = patient.rows();
        for (int j = 0; j < h - 5; ) {
            for (int k = 0; k < w - 5; ) {
                double sum = 0.0;
                for (int t = k; t < k + 5; t++) {
                    for (int i = j; i < j + 5; i++) {
                        double[] data = patient.get(i, t);
                        sum += data[0];
                    }
                }
                if (sum < 255 * 19) {
                    rectangle(patient, new Point(k, j), new Point(k + 4, j + 4), new Scalar(0, 0, 0), -1);
                }
                k = k + 5;
            }
            j = j + 5;
        }

        imwrite("/sdcard/MathRecogData/__algo__" + debug, patient);
        Mat symbolsCont = patient; // we are going to extract symbols from this matrix
        // smooth the image
        GaussianBlur(patient, patient, new Size(5, 5), 0);
        imwrite("/sdcard/MathRecogData/__gb__" + debug, patient);

        // find borders
        Canny(patient, patient, 200.0, 295.0, 3, true);

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
}
