package org.opencv.samples.mathrecog;

import java.io.FileOutputStream;
import java.util.List;

import org.opencv.android.JavaCameraView;
import org.opencv.core.Mat;

import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.Size;
import android.util.AttributeSet;
import android.util.Log;

import static org.opencv.highgui.Highgui.imwrite;

public class CameraView extends JavaCameraView implements PictureCallback {

    private String mPictureFileName;

    public CameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public List<Size> getResolutionList() {
        if (mCamera == null) {
            initializeCamera(1920, 1024);
        }
        return mCamera.getParameters().getSupportedPreviewSizes();
    }

    public void setResolution(Size resolution) {
        disconnectCamera();
        mMaxHeight = resolution.height;
        mMaxWidth = resolution.width;
        connectCamera(getWidth(), getHeight());
    }

    public Size getResolution() {
        return mCamera.getParameters().getPreviewSize();
    }

    public void takePicture(final String fileName) {
        this.mPictureFileName = fileName;
        // Postview and jpeg are sent in the same buffers if the queue is not empty when performing a capture.
        // Clear up buffers to avoid mCamera.takePicture to be stuck because of a memory issue
        mCamera.setPreviewCallback(null);

        // PictureCallback is implemented by the current class
        mCamera.takePicture(null, null, this);
    }

    @Override
    public void onPictureTaken(byte[] data, Camera camera) {
        // The camera preview was automatically stopped. Start it again.
        mCamera.startPreview();
        mCamera.setPreviewCallback(this);

        // Write the image in a file (in jpeg format)
        try {
            FileOutputStream fos = new FileOutputStream(mPictureFileName);

            fos.write(data);
            fos.close();
            ImageProcessor imp = new ImageProcessor(mPictureFileName);
            imp.Process();


        } catch (java.io.IOException e) {
            Log.e("PictureDemo", "Exception in photoCallback", e);
        }

    }
}
