"""OpenStreetMap API integration module."""

from .client import OSMClient, OSMClientError, Coordinates

__all__ = ['OSMClient', 'OSMClientError', 'Coordinates']
