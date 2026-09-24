"""Read-only macOS IOReport probe; private ABI, diagnostic use only.

Print raw channel state residency, without assuming the state index is MHz.
Run in a separate process. No model, driver reset or power setting changes.
"""
import argparse
import ctypes as C
import json
import plistlib
import time


class IOReport:
    def __init__(self, group, subgroup):
        self.cf = C.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        self.io = C.CDLL('/usr/lib/libIOReport.dylib')
        p = C.c_void_p

        def bind(lib, name, result, args):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = result, args
            return fn

        self.release = bind(self.cf, 'CFRelease', None, [p])
        string = bind(self.cf, 'CFStringCreateWithCString', p, [p, C.c_char_p, C.c_uint32])
        copy = bind(self.io, 'IOReportCopyChannelsInGroup', p,
                    [p, p, C.c_uint64, C.c_uint64, C.c_uint64])
        mutable = bind(self.cf, 'CFDictionaryCreateMutableCopy', p, [p, C.c_long, p])
        subscribe = bind(self.io, 'IOReportCreateSubscription', p,
                         [p, p, C.POINTER(p), C.c_uint64, p])
        self.sample = bind(self.io, 'IOReportCreateSamples', p, [p, p, p])
        self.serialize = bind(self.cf, 'CFPropertyListCreateData', p,
                              [p, p, C.c_long, C.c_ulong, C.POINTER(p)])
        self.length = bind(self.cf, 'CFDataGetLength', C.c_long, [p])
        self.bytes = bind(self.cf, 'CFDataGetBytePtr', p, [p])
        self.key = string(None, b'IOReportChannels', 0x08000100)
        self.get = bind(self.cf, 'CFDictionaryGetValue', p, [p, p])
        self.count = bind(self.cf, 'CFArrayGetCount', C.c_long, [p])
        self.at = bind(self.cf, 'CFArrayGetValueAtIndex', p, [p, C.c_long])
        self.cstring = bind(self.cf, 'CFStringGetCString', C.c_bool,
                            [p, p, C.c_long, C.c_uint32])
        self.states = bind(self.io, 'IOReportStateGetCount', C.c_int, [p])
        self.state_name = bind(self.io, 'IOReportStateGetNameForIndex', p, [p, C.c_int])
        self.residency = bind(self.io, 'IOReportStateGetResidency', C.c_uint64, [p, C.c_int])
        self.transitions = bind(self.io, 'IOReportStateGetInTransitions', C.c_uint64, [p, C.c_int])
        self.format = bind(self.io, 'IOReportChannelGetFormat', C.c_int, [p])
        self.integer = bind(self.io, 'IOReportSimpleGetIntegerValue', C.c_int64, [p, C.c_int])
        g = string(None, group.encode(), 0x08000100)
        s = string(None, subgroup.encode(), 0x08000100) if subgroup else None
        channels = copy(g, s, 0, 0, 0)
        self.release(g)
        if s:
            self.release(s)
        if not channels:
            raise RuntimeError('no channels returned')
        self.request = mutable(None, 0, channels)
        self.release(channels)
        self.channels = p()
        self.subscription = subscribe(None, self.request, C.byref(self.channels), 0, None)
        if not self.subscription or not self.channels.value:
            raise RuntimeError('IOReport subscription unavailable with current permissions/ABI')

    def read(self):
        sample = self.sample(self.subscription, self.channels, None)
        if not sample:
            raise RuntimeError('IOReport sample unavailable')
        error = C.c_void_p()
        data = self.serialize(None, sample, 100, 0, C.byref(error))
        try:
            if not data:
                raise RuntimeError('sample is not a serializable property list')
            result = plistlib.loads(C.string_at(self.bytes(data), self.length(data)))
            channels = self.get(sample, self.key)
            for i, channel in enumerate(result.get('IOReportChannels', [])):
                native = self.at(channels, i)
                format_id = self.format(native)
                channel['format_id'] = format_id
                if format_id == 1:
                    channel['integer_raw'] = self.integer(native, 0)
                if format_id != 2:
                    continue
                n = self.states(native)
                if not 0 <= n <= 256:
                    raise RuntimeError(f'unexpected state count {n}; private ABI may differ')
                decoded = []
                for state in range(n):
                    name = self.state_name(native, state)
                    buffer = C.create_string_buffer(256)
                    label = None
                    if name and self.cstring(name, buffer, 256, 0x08000100):
                        label = buffer.value.decode()
                    decoded.append({'index': state, 'name': label,
                                    'residency_raw': self.residency(native, state),
                                    'in_transitions': self.transitions(native, state)})
                channel['decoded_states'] = decoded
            return result
        finally:
            if data:
                self.release(data)
            if error.value:
                self.release(error)
            self.release(sample)

    def close(self):
        # Subscription is an opaque IOReport object, not asserted to be a CF
        # object. Process exit releases it; do not CFRelease an unknown type.
        self.release(self.channels)
        self.release(self.request)
        self.release(self.key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', default='GPU Stats')
    parser.add_argument('--subgroup', default='GPU Performance States')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--interval', type=float, default=1)
    args = parser.parse_args()
    if not 1 <= args.samples <= 120 or not 0.1 <= args.interval <= 5:
        parser.error('samples must be 1..120 and interval 0.1..5 seconds')
    probe = IOReport(args.group, args.subgroup)
    try:
        for i in range(args.samples):
            if i:
                time.sleep(args.interval)
            start = time.monotonic()
            data = probe.read()
            print(json.dumps({'sample': i, 'monotonic': start,
                              'read_seconds': time.monotonic()-start,
                              'raw': data}, default=lambda v: {'bytes_hex': v.hex()}), flush=True)
    finally:
        probe.close()


if __name__ == '__main__':
    main()
