import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [ads, setAds] = useState([]);

  useEffect(() => {
    // Fetch ad recommendations when the component mounts
    axios.get('/api/get-ads')  
      .then(response => setAds(response.data))
      .catch(error => console.error('Error fetching ads', error));
  }, []);

  const handleAdClick = (adId) => {
    // Send click data to the backend
    axios.post('/api/capture-click', { adId })
      .catch(error => console.error('Error sending click data', error));

    // You might also redirect the user to the ad's link or perform other actions
  };

  return (
    <div className="App">
      <h1>Recommended Ads</h1>
      <div className="ad-container">
        {ads.map(ad => (
          <div key={ad.id} className="ad" onClick={() => handleAdClick(ad.id)}>
            <img src={ad.imageUrl} alt={ad.title} />
            <h3>{ad.title}</h3>
            <p>{ad.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
