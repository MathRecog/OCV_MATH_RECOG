package org.opencv.samples.mathrecog;

import android.os.Environment;

import org.opencv.core.*;
import static org.opencv.highgui.Highgui.*;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by VladimirGerasimov on 05.02.2015.
 */
public class ImageProcessor {
    private Mat photo;

    ImageProcessor(String img) {
        photo = imread(img, CV_LOAD_IMAGE_COLOR);
        imwrite(img.replace(".img", "_check.jpg"), photo);
        if(photo.empty()) {
            throw new Error("Cannot find the image file: " + img);
        }
    }

    public Mat toGrayscale(){
        Mat grayscale;
        cvtColor(photo, photo, COLOR_BGR2GRAY, 0);
        if(photo == null) {
            throw new Error("Cannot transform to grayscale");
        }
        return photo;
    }
}
