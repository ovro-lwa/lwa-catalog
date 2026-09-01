"""Debounced Aladin view refresh helpers (pan/zoom)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from astropy import units as u
from astropy.coordinates import SkyCoord

if TYPE_CHECKING:
    from ipyaladin import Aladin


class DebouncedAladinViewRefresh:
    """Call a callback after Aladin pan/zoom settles (ipyaladin ``_target`` / ``_fov``)."""

    def __init__(
        self,
        aladin: Aladin,
        callback: Callable[[], None],
        *,
        debounce_s: float = 0.35,
        enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._aladin = aladin
        self._callback = callback
        self._debounce_s = debounce_s
        self._enabled = enabled or (lambda: True)
        self._timer: threading.Timer | None = None
        self._handler = self._on_view_trait
        aladin.observe(self._handler, names=["_target", "_fov"])

    def cancel_pending(self) -> None:
        """Cancel a scheduled debounced refresh without detaching observers."""
        timer = self._timer
        if timer is not None:
            timer.cancel()
            self._timer = None

    def detach(self) -> None:
        """Stop observing and cancel any pending refresh."""
        self.cancel_pending()
        try:
            self._aladin.unobserve(self._handler, names=["_target", "_fov"])
        except (ValueError, KeyError, TypeError):
            pass

    def _on_view_trait(self, _change: Any) -> None:
        if not self._enabled():
            return
        timer = self._timer
        if timer is not None:
            timer.cancel()
        self._timer = threading.Timer(self._debounce_s, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        if not self._enabled():
            return
        try:
            self._callback()
        except Exception:
            # Viewers surface errors in their own status panes.
            pass


def aladin_view_center_fov(aladin: Aladin) -> tuple[SkyCoord, float]:
    """Return the current Aladin center and circular FOV in degrees."""
    target = aladin.target
    if isinstance(target, SkyCoord):
        coord = target
    else:
        ra, dec = target
        coord = SkyCoord(ra=ra, dec=dec, frame="icrs")
    fov = float(aladin.fov.to(u.deg).value)
    return coord, fov
