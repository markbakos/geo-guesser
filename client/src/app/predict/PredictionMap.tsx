"use client"

import { useEffect, useRef } from "react"

const L = typeof window !== "undefined" ? require("leaflet") : null
import "leaflet/dist/leaflet.css"

const DefaultIcon = L ?
    L.icon({
    iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
    iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
    shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
    })
    : null

if (L) {
    L.Marker.prototype.options.icon = DefaultIcon
}

interface PredictionMapProps {
    prediction: [number, number] | null
}

export default function PredictionMap({ prediction }: PredictionMapProps) {
    const mapRef = useRef<L.Map | null>(null)
    const markerRef = useRef<L.Marker | null>(null)
    const mapContainerRef = useRef<HTMLDivElement | null>(null)

    useEffect(() => {
        if (mapContainerRef.current && !mapRef.current) {
            mapRef.current = L.map(mapContainerRef.current).setView([0, 0], 2)
            L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(mapRef.current)
        }

        return () => {
            mapRef.current?.remove()
            mapRef.current = null
        }
    }, [])

    useEffect(() => {
        if (prediction && mapRef.current) {
            const [lat, lng] = prediction
            const newLatLng = new L.LatLng(lat, lng)

            if (markerRef.current) {
                markerRef.current.setLatLng(newLatLng)
            } else {
                markerRef.current = L.marker(newLatLng).addTo(mapRef.current!)
            }

            mapRef.current.setView(newLatLng, 8)
        }
    }, [prediction])

    return <div ref={mapContainerRef} className="h-full min-h-[400px] rounded-lg shadow-md" />
}