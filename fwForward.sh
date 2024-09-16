# https://monovm.com/blog/port-forwarding-in-windows
# netsh interface portproxy add v4tov4 listenport=80 listenaddress=0.0.0.0 connectport=8080 connectaddress=192.168.1.10
# netsh interface portproxy delete v4tov4 listenport= listenaddress=
echo digitare il seguente comando in una shell di amministratore \(windows\)
echo netsh.exe interface portproxy set v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8051 connectaddress=$(wsl.exe hostname -I)
echo netsh interface portproxy show all
