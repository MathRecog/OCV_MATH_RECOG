package org.opencv.samples.mathrecog;

import android.os.Environment;

import org.opencv.core.*;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.highgui.Highgui.*;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by VladimirGerasimov on 05.02.2015.
 */
public class ImageProcessor {
    private Mat photo;
    String debug;
    ImageProcessor(String img) {
        photo = imread(img, CV_LOAD_IMAGE_COLOR);
        if(photo.empty()) {
            throw new Error("Cannot find the image file: " + img);
        }
        File f = new File(img);
        debug = f.getName();
        debug.replace(".jpg", "_test.jpg");
    }

    public void Process(){
        Mat patient = photo;
        cvtColor(patient, patient, COLOR_BGR2GRAY, 0);
        Size ksize = new Size(4,4);
        GaussianBlur(patient, patient, ksize, 5);
        adaptiveThreshold(patient, patient, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 35);
        Canny(patient, patient, 20.0, 30.0);
        Mat hierarchy = new Mat();
        List<MatOfPoint> conts = new ArrayList<MatOfPoint>();
        findContours(patient, conts, hierarchy, 3, 1);

        imwrite("/sdcard/MathRecogData/" + debug, patient);

    }
}
