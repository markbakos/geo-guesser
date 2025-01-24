"use client"

import { useEffect, useRef } from "react"
import L from "leaflet"
import "leaflet/dist/leaflet.css"

interface PredictionMapProps {
    prediction: [number, number] | null
}

export default function PredictionMap({ prediction }: PredictionMapProps) {
    const mapRef = useRef<L.Map | null>(null)
    const markerRef = useRef<L.Marker | null>(null)

    useEffect(() => {
        if (!mapRef.current) {
            mapRef.current = L.map("map").setView([0, 0], 2)
            L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(mapRef.current)
        }

        return () => {
            if (mapRef.current) {
                mapRef.current.remove()
                mapRef.current = null
            }
        }
    }, [])

    useEffect(() => {
        if (prediction && mapRef.current) {
            const [lat, lng] = prediction

            if (markerRef.current) {
                markerRef.current.setLatLng([lat, lng])
            } else {
                markerRef.current = L.marker([lat, lng]).addTo(mapRef.current)
            }

            mapRef.current.setView([lat, lng], 8)
        }
    }, [prediction])

    return <div id="map" className="h-full min-h-[400px] rounded-lg shadow-md" />
}