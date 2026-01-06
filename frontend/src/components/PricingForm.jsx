import React, { useState } from "react";
import "./PricingForm.css";

function PricingForm() {
  const [formData, setFormData] = useState({
    price: "",
    stock_level: "",
    day_of_week: "",
    is_weekend: "",
    month: ""
  });

  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setErrors({ ...errors, [e.target.name]: "" });
  };

  // 🧾 Validation
  const validate = () => {
    let newErrors = {};

    if (!formData.price || formData.price <= 0)
      newErrors.price = "Enter a valid price (> 0)";

    if (!formData.stock_level || formData.stock_level < 0)
      newErrors.stock_level = "Stock level cannot be negative";

    if (formData.day_of_week < 0 || formData.day_of_week > 6)
      newErrors.day_of_week = "Day must be between 0 and 6";

    if (formData.is_weekend !== "0" && formData.is_weekend !== "1")
      newErrors.is_weekend = "Enter 1 (Yes) or 0 (No)";

    if (formData.month < 1 || formData.month > 12)
      newErrors.month = "Month must be between 1 and 12";

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const predictPrice = async () => {
    if (!validate()) return;

    setLoading(true);
    setResult(null);
    setProgress(0);

    const interval = setInterval(() => {
      setProgress((prev) => (prev < 90 ? prev + 10 : prev));
    }, 300);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict-price", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          price: Number(formData.price),
          stock_level: Number(formData.stock_level),
          day_of_week: Number(formData.day_of_week),
          is_weekend: Number(formData.is_weekend),
          month: Number(formData.month)
        })
      });

      const data = await response.json();

      setTimeout(() => {
        clearInterval(interval);
        setProgress(100);
        setResult(data);
        setLoading(false);
      }, 3000);

    } catch {
      clearInterval(interval);
      alert("API error. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>📦 PriceOptima</h2>
      <p className="subtitle">ML-Based Dynamic Pricing Dashboard</p>

      {[
        ["price", "💰 Product Price", "Example: 49.99"],
        ["stock_level", "📦 Stock Level", "Example: 1200"],
        ["day_of_week", "📅 Day of Week (0–6)", "Example: 6 (Sunday)"],
        ["is_weekend", "🗓️ Is Weekend (0 or 1)", "Example: 1 = Yes"],
        ["month", "📆 Month (1–12)", "Example: 12"]
      ].map(([name, label, placeholder]) => (
        <div className="field" key={name}>
          <label>{label}</label>
          <input
            name={name}
            placeholder={placeholder}
            onChange={handleChange}
            className={errors[name] ? "input-error" : ""}
          />
          {errors[name] && <span className="error">{errors[name]}</span>}
        </div>
      ))}

      <button onClick={predictPrice} disabled={loading}>
        {loading ? "Analyzing..." : "🚀 Get Recommended Price"}
      </button>

      {loading && (
        <div className="progress-container">
          <div className="progress-bar" style={{ width: `${progress}%` }} />
          <p className="progress-text">Analyzing demand & pricing...</p>
        </div>
      )}

      {!loading && result && (
        <div className="result">
          <p>📈 <b>Predicted Demand:</b> {result.predicted_demand}</p>
          <p>💸 <b>Recommended Price:</b> ₹{result.recommended_price}</p>
        </div>
      )}
    </div>
  );
}

export default PricingForm;
