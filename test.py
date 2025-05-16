import requests
cert_path = "/Users/momin.naqvi/Downloads/ZscalerRootCertificate-2048-SHA256.pem"
response = requests.get("https://google.com", verify=cert_path)
print(response.status_code) # Should return 200 if successful