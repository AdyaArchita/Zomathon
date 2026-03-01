import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { MapPin, Search, ChevronDown, Filter, Star, Clock, ChevronRight } from 'lucide-react';

const DiningPage = ({ location }) => {
    const [restaurants, setRestaurants] = useState([]);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    // 1. Your 20 Local Assets List
    const localImages = [
        'Barbeque Nation', 'Bikanervala', 'Din Tai Fung', 'Empire Restaurant',
        "Haldiram's", 'Hard Rock Cafe', 'Heat - Edsa Shangri-La',
        'Izakaya Kikufuji', "Karim's", 'The Little Breath', 'Nobu', 'Ooma',
        'Paradise Biryani', 'Punjab Grill', 'Sambo Kojin', 'Saravana Bhavan',
        'Social', 'The Coffee Bean', 'Wildfire', 'Wow! Momo'
    ];

    useEffect(() => {
        setLoading(true);
        fetch(`http://localhost:8000/restaurants/location/${encodeURIComponent(location)}`)
            .then(res => res.json())
            .then(data => {
                // MIXED LOGIC: Filter for local assets AND match the current city
                const filtered = data.filter(res => {
                    const isFeatured = localImages.includes(res.restaurant_name?.trim());
                    const resLoc = res.locality?.toLowerCase() || "";
                    const userCity = location?.toLowerCase() || "";
                    return isFeatured && resLoc.includes(userCity);
                });
                setRestaurants(filtered);
                setLoading(false);
            })
            .catch(() => setLoading(false));
    }, [location]);

    const collections = [
        { title: 'New in Town', places: 12, image: 'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=300&q=80' },
        { title: 'Romantic Dining', places: 8, image: 'https://images.unsplash.com/photo-1544148103-0773bf10d330?auto=format&fit=crop&w=300&q=80' },
        { title: 'Great Buffets', places: 15, image: 'https://images.unsplash.com/photo-1555939594-58d7cb561ad1?auto=format&fit=crop&w=300&q=80' },
    ];

    return (
        <div className="bg-white min-h-screen pb-32 text-left font-sans">
            {/* Header synced with App location */}
            <div className="px-4 pt-6 pb-4 sticky top-0 bg-white z-50 border-b border-gray-50">
                <div className="flex items-center space-x-2">
                    <MapPin size={22} className="text-red-500 fill-red-500" />
                    <div className="flex flex-col">
                        <div className="flex items-center">
                            <span className="text-sm font-black uppercase tracking-tight text-gray-800">{location}</span>
                            <ChevronDown size={14} className="ml-1 text-gray-400" />
                        </div>
                        <span className="text-[10px] text-gray-400 font-bold uppercase tracking-widest leading-none mt-0.5">Dining Out</span>
                    </div>
                </div>
            </div>

            {/* Search with modern styling */}
            <div className="px-4 mt-6">
                <div className="bg-white border border-gray-200 rounded-2xl p-3.5 flex items-center space-x-3 shadow-sm mb-4">
                    <Search size={18} className="text-red-500" />
                    <input className="bg-transparent outline-none text-sm w-full font-bold text-gray-700" placeholder={`Search dining in ${location}...`} />
                </div>

                {/* Mixed in: Quick Filters from original version */}
                <div className="flex space-x-3 overflow-x-auto no-scrollbar pb-2">
                    <button className="flex-shrink-0 flex items-center bg-gray-50 border border-gray-100 px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest text-gray-500">
                        <Filter size={12} className="mr-1" /> Filters
                    </button>
                    <button className="flex-shrink-0 bg-gray-50 border border-gray-100 px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest text-gray-500">Rating: 4.0+</button>
                    <button className="flex-shrink-0 bg-gray-50 border border-gray-100 px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest text-gray-500">Book Table</button>
                </div>
            </div>

            {/* Handpicked Collections Section (From Earlier Version) */}
            <div className="px-4 mb-10">
                <div className="flex justify-between items-end mb-5">
                    <div>
                        <h3 className="font-black text-gray-800 text-lg uppercase tracking-tighter leading-none">Collections</h3>
                        <p className="text-[10px] text-gray-400 font-bold uppercase tracking-widest mt-1.5">Curated for {location}</p>
                    </div>
                    <span className="text-[10px] text-red-500 font-black uppercase tracking-widest">See all ▸</span>
                </div>
                <div className="flex space-x-4 overflow-x-auto no-scrollbar pb-2">
                    {collections.map((col, index) => (
                        <div key={index} className="flex-shrink-0 relative w-44 h-56 rounded-[2.5rem] overflow-hidden shadow-lg">
                            <img src={col.image} alt={col.title} loading="lazy" className="w-full h-full object-cover" />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-transparent to-transparent"></div>
                            <div className="absolute bottom-5 left-5 text-white">
                                <h4 className="font-black text-sm tracking-tight leading-tight">{col.title}</h4>
                                <p className="text-[9px] font-bold opacity-80 uppercase tracking-widest mt-1">{col.places} Places</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Trending Local Restaurants (Premium Look) */}
            <div className="px-4">
                <h3 className="font-black text-gray-800 mb-6 text-xl tracking-tighter uppercase">Trending Dining</h3>

                {loading ? (
                    <div className="space-y-8">
                        {[1, 2].map(i => <div key={i} className="h-64 bg-gray-50 rounded-[3rem] animate-pulse" />)}
                    </div>
                ) : restaurants.length > 0 ? (
                    restaurants.map((res) => (
                        <div
                            key={res.restaurant_id}
                            onClick={() => navigate(`/restaurant/${res.restaurant_id}`)}
                            className="block rounded-[3rem] border border-gray-100 overflow-hidden shadow-sm mb-12 bg-white group cursor-pointer"
                        >
                            <div className="h-60 relative overflow-hidden">
                                <img
                                    src={`/assets/Restaurants/${res.restaurant_name?.replace(/\d+$/, '').trim()}.jpg`}
                                    alt={res.restaurant_name}
                                    className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                                    loading="lazy"
                                    onError={(e) => { e.target.src = "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=800&q=80"; }}
                                />
                                <div className="absolute top-4 left-0 bg-blue-600 text-white px-4 py-1.5 text-[10px] font-black uppercase tracking-widest rounded-r-2xl shadow-lg">
                                    Flat 15% OFF
                                </div>
                                <div className="absolute inset-0 bg-gradient-to-t from-black/95 via-black/10 to-transparent opacity-90" />
                                <div className="absolute bottom-6 left-8 text-white">
                                    <h4 className="text-2xl font-black tracking-tighter leading-none mb-1">{res.restaurant_name}</h4>
                                    <p className="text-[10px] font-black opacity-80 uppercase tracking-widest">{res.locality}</p>
                                </div>
                                <div className="absolute bottom-6 right-8 bg-green-700 text-white px-2 py-1 rounded-lg flex items-center space-x-1 shadow-sm">
                                    <span className="text-[10px] font-black">4.5</span>
                                    <Star size={10} className="fill-white" />
                                </div>
                            </div>
                            <div className="p-8 flex justify-between items-center">
                                <div className="flex flex-col">
                                    <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">{res.cuisine_type}</span>
                                    <span className="text-xs font-bold text-gray-500 mt-1">₹{res.price * 2} for two</span>
                                </div>
                                <div className="flex items-center text-red-500 font-black text-[10px] uppercase tracking-widest bg-red-50 px-4 py-2 rounded-2xl">
                                    Book Table
                                </div>
                            </div>
                        </div>
                    ))
                ) : (
                    <div className="py-20 text-center bg-gray-50 rounded-[3rem] border-2 border-dashed border-gray-100">
                        <MapPin size={32} className="text-gray-200 mx-auto mb-3" />
                        <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest">No local assets found in {location}</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default DiningPage;