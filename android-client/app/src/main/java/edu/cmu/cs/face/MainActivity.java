package edu.cmu.cs.face;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkRequest;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.ImageView;

import com.google.protobuf.ByteString;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.function.Consumer;

import edu.cmu.cs.gabriel.camera.CameraCapture;
import edu.cmu.cs.gabriel.camera.ImageViewUpdater;
import edu.cmu.cs.gabriel.camera.YuvToJPEGConverter;
import edu.cmu.cs.gabriel.client.comm.ServerComm;
import edu.cmu.cs.gabriel.client.results.ErrorType;
import edu.cmu.cs.gabriel.client.results.SendSupplierResult;
import edu.cmu.cs.gabriel.protocol.Protos.InputFrame;
import edu.cmu.cs.gabriel.protocol.Protos.ResultWrapper;
import edu.cmu.cs.gabriel.protocol.Protos.PayloadType;
import edu.cmu.cs.sntp.SntpClient;


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
    private static final int confidenceThreshold = 75;

    // NTP Clock Sync
    public SntpClient sntpClient;
    private static final String NTP_SERVER = "AC-NTP0.NET.CMU.EDU";
    private Timer ntpTimer;
    private Network ntpNetwork;
    private static final long NTP_POLLING_INTERVAL = 15000;  // Request NTP clock sync every 15 sec
    private static final int NTP_BURST_POLLING_COUNT = 10;
    private static final int NTP_POLLING_TIMEOUT = 100;
    private static final long NTP_RTT_TOLERANCE = 10;
    private long minNtpRtt = NTP_POLLING_INTERVAL;
    private long lastNtpOffset = 0L;
    public boolean ntpReceived = false;
    public ConnectivityManager connectivityManager;
    private static final int APP_NETWORK_TRANSPORT_TYPE = NetworkCapabilities.TRANSPORT_WIFI;
    private static final int NTP_NETWORK_TRANSPORT_TYPE = NetworkCapabilities.TRANSPORT_WIFI;

    // Latency Profiling
    private static final int PROFILING_COUNT = 500;
    private int sentFrameCount = 0;
    private int receivedFrameCount = 0;
    private boolean doneProfiling = false;
    private FileWriter logFileWriter;
    private final SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-z", Locale.US);
    private final String LOGFILE = "Client-Timing-" + sdf.format(new Date()) + ".txt";
    public final ConcurrentLinkedDeque<String> logList = new ConcurrentLinkedDeque<>();

    private class NtpTimerTask extends TimerTask {
        @Override
        public void run() {
            // NTP burst polling
            for (int ntpi = 0; ntpi < NTP_BURST_POLLING_COUNT; ntpi++) {
                if (!sntpClient.requestTime(NTP_SERVER, NTP_POLLING_TIMEOUT, ntpNetwork)) {
                    // "labgw.elijah.cs.cmu.edu"
                    // "ec2-54-197-201-248.compute-1.amazonaws.com"
                    // "time.cloudflare.com"
                    Log.w(TAG, "Failed to request time from NTP server: Timed out after " +
                            NTP_POLLING_TIMEOUT + " milliseconds.");
                } else {
                    long rtt = sntpClient.getRoundTripTime();
                    if (rtt < minNtpRtt + NTP_RTT_TOLERANCE) {
                        lastNtpOffset = sntpClient.getNtpTime() - sntpClient.getNtpTimeReference();
                        ntpReceived = true;
                        Log.w(TAG, "Time update from NTP server using " +
                                connectivityManager.getNetworkInfo(ntpNetwork).getTypeName() +
                                ". RTT = " + sntpClient.getRoundTripTime() +
                                ", Offset = " + sntpClient.getClockOffset());
                        if (rtt < minNtpRtt) {
                            minNtpRtt = rtt;
                        }
                    }
                }
            }
        }
    }

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

        receivedFrameCount++;
        String frameRecvString = receivedFrameCount + "\tClient Recv\t" + getNetworkTimeString() + "\n";

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
        frameRecvString = frameRecvString +  receivedFrameCount + "\tClient Done\t" + getNetworkTimeString() + "\n";
        logList.add(frameRecvString);
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

    public String getNetworkTimeString() {
        if (ntpReceived) {
            long ntpNow = SystemClock.elapsedRealtime() + lastNtpOffset;
            return String.valueOf(ntpNow);
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            return String.valueOf(SystemClock.currentNetworkTimeClock().millis());
        }
        return System.currentTimeMillis() + " WARNING_MAY_NOT_USE_NETWORK_TIME_CLOCK";
    }

    private void writeLog() {
        try {
            for (String logString: logList) {
                logFileWriter.write(logString);
            }
            logFileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
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

        // Connect to network via the specific transport type
        NetworkRequest appNetworkRequest = new NetworkRequest.Builder()
                .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .addTransportType(APP_NETWORK_TRANSPORT_TYPE)
                .build();
        connectivityManager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        ConnectivityManager.NetworkCallback appNetworkCallback = new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(@NonNull Network network) {
                Log.w(TAG, "Requested app network type available, using " +
                        connectivityManager.getNetworkInfo(network).getTypeName() + " " +
                        connectivityManager.getNetworkInfo(network).getSubtypeName());
                Log.w(TAG, "Network binding for app: " + connectivityManager.bindProcessToNetwork(network));
                setupComm();
            }
        };
        connectivityManager.requestNetwork(appNetworkRequest, appNetworkCallback);

        File logFile = new File(getExternalFilesDir(null), LOGFILE);
        logFile.delete();
        logFile = new File(getExternalFilesDir(null), LOGFILE);
        try {
            logFileWriter = new FileWriter(logFile, true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        yuvToJPEGConverter = new YuvToJPEGConverter(this, 90);
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

    void setupComm() {
        NetworkRequest ntpNetworkRequest = new NetworkRequest.Builder()
                .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .addTransportType(NTP_NETWORK_TRANSPORT_TYPE)
                .build();
        ConnectivityManager.NetworkCallback ntpNetworkCallback = new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(@NonNull Network network) {
                Log.w(TAG, "Requested NTP network type available, using " +
                        connectivityManager.getNetworkInfo(network).getTypeName() + " " +
                        connectivityManager.getNetworkInfo(network).getSubtypeName());
                // Time synchronization using SNTP
                ntpNetwork = network;
                sntpClient = new SntpClient();
                ntpTimer = new Timer();
                ntpTimer.scheduleAtFixedRate(new NtpTimerTask(), 0, NTP_POLLING_INTERVAL);
            }
        };
        connectivityManager.requestNetwork(ntpNetworkRequest, ntpNetworkCallback);

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
        textToSpeech = new TextToSpeech(getApplicationContext(), onInitListener);
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
                String frameLogString = "\tClient Gen \t" + getNetworkTimeString() + "\n";
                if (receivedFrameCount >= PROFILING_COUNT) {
                    if (!doneProfiling) {
                        Log.w("PROFILE1", "Done Profiling.\n");
                        writeLog();
                        doneProfiling = true;
                    }
                    image.close();
                    return;
                }

                SendSupplierResult result = serverComm.sendSupplier(() -> {
//                    long jpegEncodingStart = SystemClock.elapsedRealtime();
                    ByteString jpegByteString = yuvToJPEGConverter.convert(image);
//                    long jpegEncodingPeriod = SystemClock.elapsedRealtime() - jpegEncodingStart;
//                    Log.w(TAG, "JPEG encoding takes " + jpegEncodingPeriod + " ms.\n");

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

                if (result == SendSupplierResult.SUCCESS) {
                    sentFrameCount++;
                    String frameSentString = sentFrameCount + frameLogString + sentFrameCount + "\tClient Send\t" +
                            getNetworkTimeString() + "\n";
                    logList.add(frameSentString);
                } else {
                    logList.add("Failed to sendSupplier: frame " + (sentFrameCount + 1) + " at" + frameLogString);
                }
            }
            // The image has either been sent or skipped. It is therefore safe to close the image.
            image.close();
        }
    };

    @Override
    protected void onPause() {
        super.onPause();
        minNtpRtt = NTP_POLLING_INTERVAL;  // Reset minNtpRtt
        if (ntpTimer != null) {
            ntpTimer.cancel();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraCapture.shutdown();
    }
}
