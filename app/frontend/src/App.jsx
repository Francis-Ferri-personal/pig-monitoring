import React, { useState, useRef, useEffect } from 'react';
import './index.css';

// Relative URLs: resolve against whatever host is serving this page.
// (In dev: Vite proxies /upload, /videos and /ws to the FastAPI backend.)
const BACKEND = '';
const WS_URL =
  (typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss://' : 'ws://') +
  (typeof window !== 'undefined' ? window.location.host : 'localhost:3000') +
  '/ws';
const STAGE_LABELS = {
  received: 'Received, waiting',
  resizing: 'Normalizing 1920x1080 @ 5fps',
  detection: 'SAM 3 detection',
  pose: 'Pose estimation (MMPose)',
  features: 'Feature extraction',
  rendering_sam: 'Rendering SAM video',
  behavior: 'Behavior inference',
  rendering_behavior: 'Rendering behavior video',
};

function App() {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const [videoUrls, setVideoUrls] = useState({
    original: null,
    keypoints: null,
    behavior: null,
    name: null,
  });

  const [activeView, setActiveView] = useState('original');

  // Video library: completed videos + in-flight uploads (id, name, status, stage)
  const [videos, setVideos] = useState([]);
  const wsRef = useRef(null);

  // Ref that always points to the upload id this frontend session started.
  const myUploadIdRef = useRef(null);
  // Keep latest videoUrls available to the websocket handler.
  const videoUrlsRef = useRef(videoUrls);
  videoUrlsRef.current = videoUrls;

  // Refs for the 3 video elements
  const originalRef = useRef(null);
  const keypointsRef = useRef(null);
  const behaviorRef = useRef(null);

  const fileInputRef = useRef(null);

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleFile = (selectedFile) => {
    if (selectedFile.type.startsWith('video/')) {
      setFile(selectedFile);
      setUploadError(null);
      // Reset URLs when a new file is uploaded
      setVideoUrls({ original: null, keypoints: null, behavior: null, name: null });
    } else {
      alert('Please upload a valid video file.');
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    setProcessing(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(`${BACKEND}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Error processing the video');
      }
      myUploadIdRef.current = data.id;
      setVideos((prev) => [...prev, {
        id: data.id,
        name: data.name,
        status: 'processing',
        stage: 'received',
        created_at: Date.now() / 1000,
      }]);
    } catch (error) {
      console.error('Error processing video:', error);
      setUploadError(error.message || 'Error connecting to the backend.');
    } finally {
      setProcessing(false);
    }
  };

  const selectVideo = (video) => {
    if (video.status !== 'completed') return;
    setVideoUrls({
      original: BACKEND + video.original_url,
      keypoints: BACKEND + video.keypoints_url,
      behavior: BACKEND + video.behavior_url,
      name: video.name,
    });
    setActiveView('original');
    setUploadError(null);
  };

  const refreshVideos = () => {
    fetch(`${BACKEND}/videos`)
      .then((res) => (res.ok ? res.json() : []))
      .then((list) =>
        setVideos((prev) => {
          const merged = new Map();
          // Existing entries first (they may hold live stage info from the websocket).
          prev.forEach((v) => merged.set(v.id, v));
          list.forEach((v) => merged.set(v.id, v));
          return [...merged.values()];
        })
      )
      .catch(() => {});
  };

  const applyVideoEvent = (video) => {
    updateLibrary(video);
    // Auto-select the video this frontend session started uploading.
    if (myUploadIdRef.current === video.id && video.status === 'completed') {
      selectVideo(video);
      myUploadIdRef.current = null;
    }
    if (myUploadIdRef.current === video.id && video.status === 'failed') {
      setUploadError(video.error || 'The backend failed to process the video.');
      myUploadIdRef.current = null;
    }
  };

  const updateLibrary = (video) => {
    setVideos((prev) => {
      const idx = prev.findIndex((v) => v.id === video.id);
      const entry = {
        id: video.id,
        name: video.name,
        status: video.status,
        stage: video.stage,
        original_url: video.original_url,
        keypoints_url: video.keypoints_url,
        behavior_url: video.behavior_url,
        error: video.error,
        created_at: video.created_at || 0,
      };
      if (idx === -1) return [...prev, entry];
      const next = [...prev];
      next[idx] = entry;
      return next;
    });
  };

  // WebSocket: live status notifications (received / processing / completed / failed).
  useEffect(() => {
    let disposed = false;
    let reconnectTimer = null;

    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        // Refresh the list when (re)connected to the backend.
        refreshVideos();
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type === 'video' && parsed.video) {
            applyVideoEvent(parsed.video);
          }
        } catch (err) {
          console.warn('Invalid websocket message', err);
        }
      };

      ws.onclose = () => {
        if (!disposed) {
          reconnectTimer = setTimeout(connect, 2000);
        }
      };
      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync logic: When switching views, copy the currentTime from the previously active video to the newly active one.
  const previousViewRef = useRef(activeView);

  useEffect(() => {
    const refs = {
      original: originalRef.current,
      keypoints: keypointsRef.current,
      behavior: behaviorRef.current,
    };

    const oldVideo = refs[previousViewRef.current];
    const newVideo = refs[activeView];

    if (oldVideo && newVideo && oldVideo !== newVideo && !isNaN(oldVideo.currentTime)) {
      newVideo.currentTime = oldVideo.currentTime;
    }
    previousViewRef.current = activeView;
  }, [activeView]);

  // Sync play/pause events across all videos so they stay perfectly in sync if they are all loaded
  const handlePlay = (e) => {
    const sourceView = e.target.dataset.view;
    if (sourceView !== activeView) return; // ignore events from hidden/programmatic videos
    const time = e.target.currentTime;

    [originalRef, keypointsRef, behaviorRef].forEach(ref => {
      if (ref.current && ref.current.dataset.view !== sourceView) {
        if (Math.abs(ref.current.currentTime - time) > 0.05) {
          ref.current.currentTime = time;
        }
        ref.current.play().catch(() => {});
      }
    });
  };

  const handlePause = (e) => {
    const sourceView = e.target.dataset.view;
    if (sourceView !== activeView) return;
    const time = e.target.currentTime;

    [originalRef, keypointsRef, behaviorRef].forEach(ref => {
      if (ref.current && ref.current.dataset.view !== sourceView) {
        ref.current.currentTime = time;
        ref.current.pause();
      }
    });
  };

  const handleSeeked = (e) => {
    const sourceView = e.target.dataset.view;
    if (sourceView !== activeView) return;
    const time = e.target.currentTime;

    [originalRef, keypointsRef, behaviorRef].forEach(ref => {
      if (ref.current && ref.current.dataset.view !== sourceView) {
        if (Math.abs(ref.current.currentTime - time) > 0.05) {
          ref.current.currentTime = time;
        }
      }
    });
  };

  const sortedVideos = [...videos].sort((a, b) =>
    (b.created_at || 0) - (a.created_at || 0) || String(b.id).localeCompare(String(a.id))
  );

  return (
    <div className="container">
      <header>
        <h1>Video Inference Comparison</h1>
      </header>

      <div
        className={`upload-container ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleChange}
          style={{ display: 'none' }}
        />
        {file ? (
          <div className="upload-selected">
            <span className="upload-icon-small">🎬</span>
            <span className="upload-filename">{file.name}</span>
          </div>
        ) : (
          <div className="upload-empty">
            <div className="upload-icon">📁</div>
            <div className="upload-text">
              Drag and drop a video here, or click to select
            </div>
          </div>
        )}
      </div>

      <div className="player-area">
        <div className="sidebar">
          <div className="sidebar-title">Views</div>
          <button
            className={`view-btn ${activeView === 'original' ? 'active' : ''}`}
            onClick={() => setActiveView('original')}
            disabled={!videoUrls.original}
          >
            Original
          </button>
          <button
            className={`view-btn ${activeView === 'keypoints' ? 'active' : ''}`}
            onClick={() => setActiveView('keypoints')}
            disabled={!videoUrls.original}
          >
            Keypoints
          </button>
          <button
            className={`view-btn ${activeView === 'behavior' ? 'active' : ''}`}
            onClick={() => setActiveView('behavior')}
            disabled={!videoUrls.original}
          >
            Behavior
          </button>

          <div className="sidebar-title library-title">Processed</div>
          <div className="video-library">
            {sortedVideos.length === 0 ? (
              <div className="library-empty">No processed videos</div>
            ) : (
              sortedVideos.map((v) => (
                <button
                  key={v.id}
                  className={`library-item ${v.status === 'processing' || v.status === 'received' ? 'library-processing' : ''}`}
                  onClick={() => v.status === 'completed' && selectVideo(v)}
                  disabled={v.status !== 'completed'}
                  title={v.error || undefined}
                >
                  <span className="library-id">#{v.id}</span>
                  <span className="library-name">{v.name}</span>
                  {v.status !== 'completed' ? (
                    <span className="library-status">{STAGE_LABELS[v.stage] || v.status}</span>
                  ) : (
                    <span className="library-status library-done">Ready</span>
                  )}
                </button>
              ))
            )}
          </div>

          <button
            className="process-btn"
            onClick={handleProcess}
            disabled={!file || processing}
          >
            {processing ? 'Uploading...' : 'Process Video'}
          </button>
          {uploadError && <div className="upload-error">{uploadError}</div>}
        </div>

        <div className="video-container">
          {!videoUrls.original ? (
            <div className="placeholder-video">
              <span style={{fontSize: '3rem'}}>🎥</span>
              <span>No video processed yet</span>
            </div>
          ) : (
            <>
              {/* Render all 3 videos but only show the active one. */}
              <video
                ref={originalRef}
                src={videoUrls.original}
                className={activeView === 'original' ? '' : 'hidden-video'}
                controls={true}
                onPlay={handlePlay}
                onPause={handlePause}
                onSeeked={handleSeeked}
                data-view="original"
              />
              <video
                ref={keypointsRef}
                src={videoUrls.keypoints}
                className={activeView === 'keypoints' ? '' : 'hidden-video'}
                controls={true}
                onPlay={handlePlay}
                onPause={handlePause}
                onSeeked={handleSeeked}
                data-view="keypoints"
              />
              <video
                ref={behaviorRef}
                src={videoUrls.behavior}
                className={activeView === 'behavior' ? '' : 'hidden-video'}
                controls={true}
                onPlay={handlePlay}
                onPause={handlePause}
                onSeeked={handleSeeked}
                data-view="behavior"
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
