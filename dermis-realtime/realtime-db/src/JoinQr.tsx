import { useEffect, useState } from 'react';
import QRCode from 'qrcode';

export default function JoinQr({ href }: { href: string }) {
  const [image, setImage] = useState('');
  useEffect(() => {
    let live = true;
    void QRCode.toDataURL(href, { width: 172, margin: 1, color: { dark: '#07111a', light: '#e5fbff' } })
      .then(data => { if (live) setImage(data); });
    return () => { live = false; };
  }, [href]);
  return <a className="join-qr" href={href} title="Open room link">{image && <img src={image} alt="QR code to join this shared room" />}</a>;
}
