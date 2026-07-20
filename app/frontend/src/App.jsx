import React, { useState, useRef, useEffect } from 'react';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [processing, setProcessing] = useState(false);
  
  // URLs for the 3 videos
  const [videoUrls, setVideoUrls] = useState({
    original: null,
    keypoints: null,
    behavior: null
  });
  
  const [activeView, setActiveView] = useState('original');
  
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
      // Reset URLs when a new file is uploaded
      setVideoUrls({
        original: null,
        keypoints: null,
        behavior: null
      });
    } else {
      alert('Please upload a valid video file.');
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    setProcessing(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Assuming backend is running on localhost:8008
      const response = await fetch('http://localhost:8008/upload', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      // Update URLs with the response from the backend
      const baseUrl = 'http://localhost:8008';
      setVideoUrls({
        original: baseUrl + data.original_url,
        keypoints: baseUrl + data.keypoints_url,
        behavior: baseUrl + data.behavior_url
      });
      setActiveView('original');
    } catch (error) {
      console.error('Error processing video:', error);
      alert('Error connecting to the backend. Is the server running?');
    } finally {
      setProcessing(false);
    }
  };

  // Sync logic: When switching views, copy the currentTime from the previously active video to the newly active one.
  const previousViewRef = useRef(activeView);
  
  useEffect(() => {
    const refs = {
      original: originalRef.current,
      keypoints: keypointsRef.current,
      behavior: behaviorRef.current
    };
    
    const oldVideo = refs[previousViewRef.current];
    const newVideo = refs[activeView];
    
    if (oldVideo && newVideo && oldVideo !== newVideo && !isNaN(oldVideo.currentTime)) {
      newVideo.currentTime = oldVideo.currentTime;
      if (!oldVideo.paused) {
        newVideo.play().catch(e => console.error("Playback failed:", e));
      } else {
        newVideo.pause();
      }
    }
    previousViewRef.current = activeView;
  }, [activeView]);

  // Sync play/pause events across all videos so they stay perfectly in sync if they are all loaded
  const handlePlay = (e) => {
    const sourceView = e.target.dataset.view;
    const time = e.target.currentTime;
    
    [originalRef, keypointsRef, behaviorRef].forEach(ref => {
      if (ref.current && ref.current.dataset.view !== sourceView) {
        ref.current.currentTime = time;
        ref.current.play().catch(err => console.log(err));
      }
    });
  };

  const handlePause = (e) => {
    const sourceView = e.target.dataset.view;
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
    const time = e.target.currentTime;
    
    [originalRef, keypointsRef, behaviorRef].forEach(ref => {
      if (ref.current && ref.current.dataset.view !== sourceView) {
        ref.current.currentTime = time;
      }
    });
  };

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
          >
            Video original
          </button>
          <button 
            className={`view-btn ${activeView === 'keypoints' ? 'active' : ''}`}
            onClick={() => setActiveView('keypoints')}
          >
            Video con keypoints
          </button>
          <button 
            className={`view-btn ${activeView === 'behavior' ? 'active' : ''}`}
            onClick={() => setActiveView('behavior')}
          >
            Behavior
          </button>
          
          <button 
            className="process-btn" 
            onClick={handleProcess}
            disabled={!file || processing}
          >
            {processing ? 'Processing...' : 'Process Video'}
          </button>
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
                controls={activeView === 'original'}
                onPlay={handlePlay}
                onPause={handlePause}
                onSeeked={handleSeeked}
                data-view="original"
              />
              <video
                ref={keypointsRef}
                src={videoUrls.keypoints}
                className={activeView === 'keypoints' ? '' : 'hidden-video'}
                controls={activeView === 'keypoints'}
                onPlay={handlePlay}
                onPause={handlePause}
                onSeeked={handleSeeked}
                data-view="keypoints"
              />
              <video
                ref={behaviorRef}
                src={videoUrls.behavior}
                className={activeView === 'behavior' ? '' : 'hidden-video'}
                controls={activeView === 'behavior'}
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
