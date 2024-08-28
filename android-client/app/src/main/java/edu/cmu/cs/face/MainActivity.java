package edu.cmu.cs.face;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.Consumer;

import edu.cmu.cs.gabriel.camera.CameraCapture;
import edu.cmu.cs.gabriel.camera.ImageViewUpdater;
import edu.cmu.cs.gabriel.camera.YuvToJPEGConverter;
import edu.cmu.cs.gabriel.client.comm.ServerComm;
import edu.cmu.cs.gabriel.client.results.ErrorType;
import edu.cmu.cs.gabriel.protocol.Protos.InputFrame;
import edu.cmu.cs.gabriel.protocol.Protos.ResultWrapper;
import edu.cmu.cs.gabriel.protocol.Protos.PayloadType;


public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final String SOURCE = "openface";
    private static final int PORT = 9099;
    private static final int WIDTH = 1280;  // (320, 640, 1280, 1920)
    private static final int HEIGHT = 720;  // (240, 480, 720, 1080)
    private ServerComm serverComm;
    private YuvToJPEGConverter yuvToJPEGConverter;
    private CameraCapture cameraCapture;
    private TextToSpeech textToSpeech;
    private ImageViewUpdater annotationViewUpdater;
    private boolean readyForServer = false;
    private boolean useBackCamera = true;
    private HashMap<String, Long> faceRecognized = new HashMap<>();
    private static final long nameTTSCoolDownTime = 3 * 60 * 1000L;
    private static final int confidenceThreshold = 70;

    private final Consumer<ResultWrapper> consumer = resultWrapper -> {
//        try {
//            ToClientExtras toClientExtras = ToClientExtras.parseFrom(
//                    resultWrapper.getExtras().getValue());
//            step = toClientExtras.getStep();
//            if (step.equals(WCA_FSM_END) && !logCompleted) {
//                readyForServer = false;
//            }
//        } catch (InvalidProtocolBufferException e) {
//            Log.e(TAG, "Protobuf parse error", e);
//        }

            if (resultWrapper.getResultsCount() == 0) {
                Log.w(TAG, "Server returned empty results.");
                return;
            }

            // Load the user guidance (audio, image/video) from the result wrapper
            for (ResultWrapper.Result result : resultWrapper.getResultsList()) {
                if (result.getPayloadType() == PayloadType.IMAGE) {
                    ByteString annotatedJpeg = result.getPayload();
                    annotationViewUpdater.accept(annotatedJpeg);
                } else if (result.getPayloadType() == PayloadType.TEXT) {
                    ByteString faceResults = result.getPayload();
                    String faceResultsString = faceResults.toStringUtf8();
                    speakOutNames(faceResultsString);
                }
            }
    };

    private void speakOutNames(String faceResultsString) {
        String[] facesStrings = faceResultsString.split(",");
        List<String> namesToSpeakOut = new ArrayList<>();
        long currentTime = System.currentTimeMillis();
        for (String faceString: facesStrings) {
            String[] faceResult = faceString.split(": ");
            String name = faceResult[0];
            int confidence = Integer.parseInt(faceResult[1].replace("%", ""));
            Log.i(TAG, "Detected: [" + name + "] " + confidence + "%");

            // Do not repeat the same names within <nameTTSCoolDownTime>
            if (confidence > confidenceThreshold) {
                if (!faceRecognized.containsKey(name) ||
                        currentTime - faceRecognized.get(name) > nameTTSCoolDownTime) {
                    faceRecognized.put(name, currentTime);
                    namesToSpeakOut.add(name);
                }
            }
        }
        if (namesToSpeakOut.size() > 0) {
            String speech = "Probably: " + String.join(" and ", namesToSpeakOut);
            this.textToSpeech.speak(speech, TextToSpeech.QUEUE_ADD, null, null);
            Log.i(TAG, "Saying: " + speech);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ImageView annotationView = findViewById(R.id.annotationView);
        annotationViewUpdater = new ImageViewUpdater(annotationView);
        PreviewView viewFinder = findViewById(R.id.viewFinder);
        ImageView camButton = findViewById(R.id.imgSwitchCam);

        Consumer<ErrorType> onDisconnect = errorType -> {
            Log.e(TAG, "Disconnect Error: " + errorType.name());
            finish();
        };
        serverComm = ServerComm.createServerComm(
                consumer, BuildConfig.GABRIEL_HOST, PORT, getApplication(), onDisconnect);

        TextToSpeech.OnInitListener onInitListener = status -> {
            if (status == TextToSpeech.ERROR) {
                Log.e(TAG, "TextToSpeech initialization failed with status " + status);
            }
//            ToServerExtras toServerExtras = ToServerExtras.newBuilder().setStep(step).build();
            InputFrame inputFrame = InputFrame.newBuilder()
//                    .setExtras(pack(toServerExtras))
                    .build();

            // We need to wait for textToSpeech to be initialized before asking for the first
            // instruction.
            serverComm.send(inputFrame, SOURCE, /* wait */ true);
            readyForServer = true;
        };
        this.textToSpeech = new TextToSpeech(getApplicationContext(), onInitListener);
        yuvToJPEGConverter = new YuvToJPEGConverter(this, 100);

        camButton.setOnClickListener(v -> {
            if (useBackCamera) {
                cameraCapture = new CameraCapture(
                        MainActivity.this, analyzer, WIDTH, HEIGHT, viewFinder,
                        CameraSelector.DEFAULT_FRONT_CAMERA, false);
                useBackCamera = false;
            } else {
                cameraCapture = new CameraCapture(
                        MainActivity.this, analyzer, WIDTH, HEIGHT, viewFinder,
                        CameraSelector.DEFAULT_BACK_CAMERA, false);
                useBackCamera = true;
            }
        });

        cameraCapture = new CameraCapture(this, analyzer, WIDTH, HEIGHT, viewFinder,
                CameraSelector.DEFAULT_BACK_CAMERA, false);
    }

    // Based on
    // https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/compiler/java/java_message.cc#L1387
//    public static Any pack(ToServerExtras toServerExtras) {
//        return Any.newBuilder()
//                .setTypeUrl("type.googleapis.com/wca.ToServerExtras")
//                .setValue(toServerExtras.toByteString())
//                .build();
//    }

    final private ImageAnalysis.Analyzer analyzer = new ImageAnalysis.Analyzer() {
        @Override
        public void analyze(@NonNull ImageProxy image) {
            if (readyForServer) {
                serverComm.sendSupplier(() -> {
                    ByteString jpegByteString = yuvToJPEGConverter.convert(image);

//                ToServerExtras toServerExtras = ToServerExtras.newBuilder()
//                        .setStep(MainActivity.this.step)
//                        .setClientCmd(clientCmd)
//                        .build();

                    return InputFrame.newBuilder()
                            .setPayloadType(PayloadType.IMAGE)
                            .addPayloads(jpegByteString)
//                        .setExtras(pack(toServerExtras))
                            .build();
                }, SOURCE, /* wait */ false);
            }
            // The image has either been sent or skipped. It is therefore safe to close the image.
            image.close();
        }
    };

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraCapture.shutdown();
    }
}
