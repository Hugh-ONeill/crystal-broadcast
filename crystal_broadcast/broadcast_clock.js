/* Presentation-clock hook for the self-hosted Showdown client.
 *
 * WHY: the sim server resolves a turn in milliseconds, but the CLIENT animates
 * it over seconds, and the gap is VARIABLE — a 6-event turn with a KO, weather
 * chip and a Tera animates far longer than a 2-move turn. Commentary paced off
 * the server's clock therefore drifts against what the viewer is actually
 * seeing. `--airi-turn-pace` only ever papered a fixed offset over a fixed
 * offset. The delay buffer needs a real presentation timeline (PTS) to schedule
 * against, and that timeline only exists inside the client.
 *
 * WHAT: wraps two Battle methods and reports both ends of each protocol line.
 *   Battle.add(cmd)  -> the line ARRIVED from the server        ("queued")
 *   Battle.run(str)  -> the line is being SHOWN to the viewer   ("presented")
 * The difference between the two timestamps IS the presentation lag, measured
 * inside one process, so no cross-process clock sync is needed.
 *
 * Lines are paired by QUEUE INDEX, not by text: protocol lines repeat
 * verbatim all the time (`|upkeep`, `|`), so text is not a key.
 *
 * `this.seeking` is null during normal playback and a turn number (or
 * Infinity) while fast-forwarding — joining a live battle replays the whole
 * backlog through run() at once. Those are not "presented" at viewer pace, so
 * they are filtered out. `preempt` runs (instantAdd, used for chat) bypass the
 * animation queue and are filtered for the same reason.
 *
 * NOTE ON SEMANTICS: run() STARTS the animation for a line; it completes some
 * time later. So "presented" is the moment the viewer begins seeing that event,
 * which is the right anchor for scheduling a commentary beat against it.
 *
 * Injected by serve_client.py into /broadcast-client.html, so the
 * pokemon-showdown-client fork stays pristine. Silently no-ops if nothing is
 * listening on the collector port.
 */
(function () {
	"use strict";
	var WS_URL = "ws://127.0.0.1:8132/";
	var MAX_PENDING = 2000;

	var ws = null, ready = false, pending = [], backoff = 500;

	function flush() {
		while (ready && pending.length) {
			try { ws.send(JSON.stringify(pending.shift())); }
			catch (e) { ready = false; return; }
		}
	}
	function send(o) {
		pending.push(o);
		if (pending.length > MAX_PENDING) pending.shift();
		flush();
	}
	function retry() {
		setTimeout(connect, backoff);
		backoff = Math.min(backoff * 2, 5000);
	}
	function connect() {
		try { ws = new WebSocket(WS_URL); }
		catch (e) { return retry(); }
		ws.onopen = function () { ready = true; backoff = 500; flush(); };
		ws.onclose = function () { ready = false; retry(); };
		ws.onerror = function () { try { ws.close(); } catch (e) {} };
	}
	connect();

	function hook() {
		if (Battle.prototype.__psClockHooked) return;
		var origAdd = Battle.prototype.add;
		var origRun = Battle.prototype.run;

		Battle.prototype.add = function (command) {
			// index this line WILL occupy once pushed
			if (command) {
				send({
					kind: "queued", id: this.id, idx: this.stepQueue.length,
					line: command, t: Date.now()
				});
			}
			return origAdd.apply(this, arguments);
		};

		Battle.prototype.run = function (str, preempt) {
			// currentStep is the index of the line being run; nextStep()
			// increments it AFTER run() returns
			var idx = this.currentStep;
			var seeking = this.seeking !== null && this.seeking !== undefined;
			var out = origRun.apply(this, arguments);
			if (str && !preempt && !seeking) {
				// read turn AFTER the run so a |turn|N line reports N
				send({
					kind: "presented", id: this.id, idx: idx, turn: this.turn,
					line: str, t: Date.now()
				});
			}
			return out;
		};

		Battle.prototype.__psClockHooked = true;
		send({ kind: "hooked", t: Date.now() });
	}

	// battle.js defines the Battle global and loads AFTER this script
	var tries = 0;
	var timer = setInterval(function () {
		if (typeof Battle === "undefined" || !Battle.prototype) {
			if (++tries > 400) clearInterval(timer);
			return;
		}
		clearInterval(timer);
		hook();
	}, 25);
})();
