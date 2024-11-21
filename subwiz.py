import subwiz

known_subdomains = ['test1.example.com', 'test2.example.com']
new_subdomains = subwiz.run(input_domains=known_subdomains, no_resolve=True)

print(new_subdomains)