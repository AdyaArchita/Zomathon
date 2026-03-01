import React, { useState, useEffect, useRef } from 'react';
import { Search, ArrowLeft, Clock, TrendingUp, X, Plus, Minus, ChevronRight } from 'lucide-react'; // Added Minus
import { useNavigate } from 'react-router-dom';
import { useCart } from '../context/CartContext';
import { getFoodImage } from '../utils/ImageHelper';

const SearchPage = () => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [isFocused, setIsFocused] = useState(false);
    const [loading, setLoading] = useState(false);
    const inputRef = useRef(null);
    const navigate = useNavigate();

    // 1. Destructure cart functions and state
    const { cart, addToCart, removeFromCart, totalAmount, totalQty } = useCart();

    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    useEffect(() => {
        if (!query.trim()) {
            setResults([]);
            return;
        }

        const delayDebounceFn = setTimeout(() => {
            setLoading(true);
            fetch(`http://localhost:8000/search?q=${query}`)
                .then(res => res.json())
                .then(data => {
                    setResults(data);
                    setLoading(false);
                })
                .catch(err => {
                    console.error("Search failed:", err);
                    setLoading(false);
                });
        }, 300);

        return () => clearTimeout(delayDebounceFn);
    }, [query]);

    const recentSearches = ['Biryani', 'Pizza', 'Burger', 'Paneer'];
    const trendingSearches = [
        { name: 'Dosa', label: 'Trending' },
        { name: 'Cakes', label: 'Popular' },
        { name: 'Chicken', label: 'Bestseller' }
    ];

    return (
        <div className="bg-white min-h-screen pb-40">
            {/* Search Header */}
            <div className="sticky top-0 bg-white z-50 px-4 pt-4 pb-3 border-b border-gray-100 shadow-sm">
                <div className="flex items-center space-x-3">
                    <button onClick={() => navigate(-1)} className="p-1 rounded-full hover:bg-gray-100 transition-colors">
                        <ArrowLeft size={20} className="text-gray-700" />
                    </button>

                    <div className={`flex-1 flex items-center bg-white border ${isFocused ? 'border-red-500 shadow-md shadow-red-100' : 'border-gray-300'} rounded-xl px-3 py-2.5 transition-all duration-200`}>
                        <Search size={18} className={`${isFocused ? 'text-red-500' : 'text-gray-400'}`} />
                        <input
                            ref={inputRef}
                            type="text"
                            className="flex-1 bg-transparent border-none outline-none ml-2 text-sm text-gray-800 placeholder-gray-400 font-medium"
                            placeholder="Search for restaurant, item or more"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onFocus={() => setIsFocused(true)}
                            onBlur={() => setIsFocused(false)}
                        />
                        {query && (
                            <button onClick={() => setQuery('')}>
                                <X size={16} className="text-gray-400" />
                            </button>
                        )}
                    </div>
                </div>
            </div>

            <div className="p-4">
                {query ? (
                    <div className="mt-4">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="font-black text-gray-400 text-[10px] uppercase tracking-widest">
                                {loading ? 'Searching...' : `Found ${results.length} items`}
                            </h3>
                        </div>

                        <div className="space-y-6">
                            {results.map((item) => {
                                // 2. Check if item is in cart for dynamic counter
                                const cartItem = cart.find((i) => String(i.item_id) === String(item.item_id));
                                const quantity = cartItem ? cartItem.qty : 0;

                                return (
                                    <div key={item.item_id} className="flex items-center justify-between group">
                                        <div className="flex items-center flex-1">
                                            <div className="w-20 h-20 bg-gray-50 rounded-2xl mr-4 overflow-hidden border border-gray-100 flex-shrink-0">
                                                <img
                                                    src={getFoodImage(item.name)}
                                                    className="w-full h-full object-cover"
                                                    alt={item.name}
                                                    loading="lazy"
                                                    onError={(e) => { e.target.src = "https://via.placeholder.com/150" }}
                                                />
                                            </div>
                                            <div className="text-left">
                                                <div className="flex items-center space-x-1.5 mb-0.5">
                                                    <div className={`w-3 h-3 border border-${item.is_veg ? 'green' : 'red'}-600 flex items-center justify-center rounded-xs`}>
                                                        <div className={`w-1.5 h-1.5 rounded-full bg-${item.is_veg ? 'green' : 'red'}-600`} />
                                                    </div>
                                                    <h4 className="font-black text-sm text-gray-800 line-clamp-1">{item.name}</h4>
                                                </div>
                                                <p className="text-[10px] text-gray-400 font-bold uppercase tracking-tight">{item.restaurant_name}</p>
                                                <p className="text-xs font-black text-gray-900 mt-1">₹{item.price}</p>
                                            </div>
                                        </div>

                                        {/* 3. DYNAMIC TOGGLE BUTTON */}
                                        <div className="shadow-lg rounded-xl overflow-hidden min-w-[100px] bg-white border border-gray-100">
                                            {quantity === 0 ? (
                                                <button
                                                    onClick={() => addToCart(item, item.restaurant_id)}
                                                    className="w-full bg-white text-green-600 font-black py-2 px-6 text-[10px] uppercase transition-all active:scale-95"
                                                >
                                                    ADD
                                                </button>
                                            ) : (
                                                <div className="flex items-center justify-between py-1.5 px-2 text-green-600">
                                                    <button
                                                        onClick={() => removeFromCart(item.item_id)}
                                                        className="p-1 hover:bg-gray-50 rounded-md active:scale-125 transition-transform"
                                                    >
                                                        <Minus size={14} strokeWidth={3} />
                                                    </button>
                                                    <span className="text-xs font-black text-gray-800 px-2">{quantity}</span>
                                                    <button
                                                        onClick={() => addToCart(item, item.restaurant_id)}
                                                        className="p-1 hover:bg-gray-50 rounded-md active:scale-125 transition-transform"
                                                    >
                                                        <Plus size={14} strokeWidth={3} />
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                ) : (
                    // ... Recent & Trending remains same
                    <>
                        <div className="mb-8 mt-2">
                            <h3 className="font-black text-gray-400 mb-4 text-[10px] uppercase tracking-widest flex items-center">
                                <Clock size={14} className="mr-2" /> Recent Searches
                            </h3>
                            <div className="flex flex-wrap gap-2">
                                {recentSearches.map((term) => (
                                    <button key={term} onClick={() => setQuery(term)} className="px-4 py-2 bg-gray-50 border border-gray-100 rounded-full text-xs font-bold text-gray-600 active:scale-95 transition-transform">
                                        {term}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div>
                            <h3 className="font-black text-gray-400 mb-4 text-[10px] uppercase tracking-widest flex items-center">
                                <TrendingUp size={14} className="mr-2 text-red-500" /> Trending near you
                            </h3>
                            <div className="grid grid-cols-2 gap-3">
                                {trendingSearches.map((trend) => (
                                    <div key={trend.name} className="border border-gray-50 bg-gray-50/30 rounded-2xl p-4 flex flex-col justify-between h-24 active:bg-white active:shadow-lg transition-all cursor-pointer" onClick={() => setQuery(trend.name)}>
                                        <span className="font-black text-sm text-gray-800">{trend.name}</span>
                                        <span className="text-[9px] text-red-500 font-black uppercase tracking-widest">{trend.label}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </>
                )}
            </div>

            {/* Sticky Cart Bar */}
            {totalQty > 0 && (
                <div className="fixed bottom-24 left-4 right-4 bg-[#E23744] text-white p-4 rounded-2xl flex justify-between items-center shadow-2xl z-50 animate-slide-up">
                    <div className="flex flex-col text-left">
                        <span className="text-[10px] font-black uppercase tracking-widest opacity-90">
                            {totalQty} ITEM{totalQty > 1 ? 'S' : ''} ADDED
                        </span>
                        <span className="text-lg font-black tracking-tighter">₹{totalAmount}</span>
                    </div>
                    <button onClick={() => navigate('/order')} className="flex items-center font-black text-sm uppercase tracking-tighter">
                        Next <ChevronRight size={18} className="ml-1" />
                    </button>
                </div>
            )}
        </div>
    );
};

export default SearchPage;