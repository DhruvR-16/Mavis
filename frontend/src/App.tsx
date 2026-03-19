import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Workout from './pages/Workout';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/workout" element={<Workout />} />
      </Routes>
    </BrowserRouter>
  );
}
