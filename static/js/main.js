window.addEventListener('load', function () {
  // Camera functionality for pages with a video element
  const video = document.getElementById('videoElement');
  if (video) {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
          video.srcObject = stream;
        })
        .catch(function (error) {
          console.error("Error accessing camera:", error);
          alert("Unable to access the camera. Please check your device settings.");
        });
    } else {
      console.error("Camera not supported in this browser.");
    }
  }
  // Clear Records functionality on attendance view page
  const clearBtn = document.getElementById('clearRecords');
  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      const tableBody = document.getElementById('attendanceTableBody');
      if (tableBody) { tableBody.innerHTML = ''; }
      const summaryDiv = document.getElementById('attendanceSummary');
      if (summaryDiv) { summaryDiv.innerHTML = ''; }
    });
  }
});
