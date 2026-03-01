import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Search, MapPin, Star, Clock, ChevronRight } from 'lucide-react';

const HomePage = ({ location = "Bhubaneswar" }) => {
  const [restaurants, setRestaurants] = useState([]);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();

  // Mapping CSV categories to Emojis
  const iconMap = {
    "Biryani": "🍲", "Pizza": "🍕", "Burger": "🍔", "Drinks": "🥤",
    "Dessert": "🍨", "Cake": "🍰", "Thali": "🍱", "Mughlai": "🍛",
    "Chinese": "🥢", "South Indian": "🥞", "Street Food": "🍢",
    "Paneer": "🧀", "Chicken": "🍗", "Dosa": "🥞", "Bread": "🍞",
    "Rice": "🍚", "Main": "🍽️", "Starter": "🥗"
  };

  useEffect(() => {
    setLoading(true);
    const fetchCats = fetch('http://localhost:8000/categories/available')
      .then(res => res.json())
      .catch(() => ["Biryani", "Pizza", "Mughlai", "Dosa", "Paneer", "Chicken", "Burgers", "Thali", "Cakes", "Dessert"]);

    const fetchRes = fetch(`http://localhost:8000/restaurants/location/${location}`)
      .then(res => res.json())
      .catch(() => [
        {
          restaurant_id: "6317637",
          restaurant_name: "Barbeque Nation",
          cuisine_type: "Mughlai • North Indian",
          location: location
        },
        {
          restaurant_id: "12345",
          restaurant_name: "The Pizza Project",
          cuisine_type: "Italian • Fast Food",
          location: location
        }
      ]);

    Promise.all([fetchCats, fetchRes]).then(([catData, resData]) => {
      // Ensure "Street Food" is in the list even if CSV is primarily "Mughlai/Chinese"
      const finalCats = catData.includes("Street Food") ? catData : ["Street Food", ...catData];

      // Deduplicate restaurants based on ID (items.csv has multiple rows per restaurant)
      const uniqueRestaurants = Array.from(
        new Map(resData.map(res => [res.restaurant_id, res])).values()
      );

      setCategories(finalCats);
      setRestaurants(uniqueRestaurants);
      setLoading(false);
    });
  }, [location]);

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) navigate(`/category/${searchQuery.toLowerCase()}`);
  };

  return (
    <div className="bg-white min-h-screen pb-24 font-sans">
      <div className="px-4 mb-8 mt-4">
        <form onSubmit={handleSearch} className="bg-white border border-gray-100 rounded-2xl p-3 flex items-center space-x-3 shadow-sm">
          <Search size={18} className="text-red-500" />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-transparent outline-none text-sm w-full font-bold text-gray-700"
            placeholder={`Search Street Food or Biryani in ${location}...`}
          />
        </form>
      </div>

      {/* --- CATEGORIES SECTION: FIX FOR CUT-OFF --- */}
      <div className="mb-10">
        <h3 className="px-4 font-black text-gray-400 mb-5 text-[10px] uppercase tracking-widest">Inspiration for your order</h3>

        {/* Added overflow-x-auto, no-scrollbar, and extra padding-right to prevent cutoff */}
        <div className="flex space-x-6 overflow-x-auto no-scrollbar px-4 pb-4">
          {categories.map(cat => (
            <Link
              key={cat}
              to={`/category/${cat.toLowerCase()}`}
              className="flex-shrink-0 text-center group active:scale-95 transition-all min-w-[64px]"
            >
              <div className="w-16 h-16 rounded-full bg-gray-50 border border-gray-100 flex items-center justify-center text-3xl shadow-sm group-hover:bg-white group-hover:shadow-md transition-all">
                {iconMap[cat] || "🍽️"}
              </div>
              <p className="text-[10px] font-black text-gray-500 mt-3 whitespace-nowrap">{cat}</p>
            </Link>
          ))}
        </div>
      </div>

      {/* --- RESTAURANTS SECTION --- */}
      <div className="px-4">
        <h3 className="font-black text-gray-800 text-lg tracking-tight mb-6">
          Explore in <span className="text-red-500">{location}</span>
        </h3>

        {loading ? (
          <div className="space-y-6">
            {[1, 2].map(i => <div key={i} className="h-64 bg-gray-50 rounded-3xl animate-pulse" />)}
          </div>
        ) : restaurants && restaurants.length > 0 ? (
          restaurants.map((res, index) => (
            <Link
              key={`${res.restaurant_id}-${index}`}
              to={`/restaurant/${res.restaurant_id}`}
              className="block rounded-[2.5rem] border border-gray-100 overflow-hidden shadow-sm active:scale-[0.98] transition-all mb-8 bg-white"
            >
              <div className="h-56 bg-gray-100 relative">
                <img
                  src={`/assets/Restaurants/${res.restaurant_name?.replace(/\d+$/, '').trim()}.jpg`}
                  alt={res.restaurant_name}
                  className="absolute inset-0 w-full h-full object-cover"
                  loading="lazy"
                  onError={(e) => {
                    e.target.src = "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=800&q=80";
                  }}
                />
                <div className="absolute top-4 left-4 bg-white/95 backdrop-blur px-3 py-1.5 rounded-xl text-[9px] font-black uppercase tracking-widest shadow-sm">
                  30-35 MINS
                </div>
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
                <div className="absolute bottom-4 left-5 text-white z-10">
                  <h4 className="text-xl font-black tracking-tight">{res.restaurant_name}</h4>
                  <p className="text-[10px] font-bold opacity-80 uppercase tracking-widest">{res.cuisine_type}</p>
                </div>
              </div>
              <div className="p-5 flex items-center justify-between">
                <div className="flex items-center text-[10px] text-gray-400 font-bold uppercase tracking-tight">
                  <Clock size={12} className="mr-1 text-gray-300" />
                  FREE DELIVERY ON ORDERS ABOVE ₹199
                </div>
                <ChevronRight size={16} className="text-gray-300" />
              </div>
            </Link>
          ))
        ) : (
          <div className="py-10 text-center bg-gray-50 rounded-[2rem] border border-gray-100 mt-4 mx-2">
            <div className="text-4xl mb-3">🍽️</div>
            <p className="text-sm font-bold text-gray-800">No restaurants found</p>
            <p className="text-[10px] text-gray-400 mt-1 uppercase tracking-widest font-black">Backend Connection Refused</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default HomePage;