#include "models/z_image/coreml_generation.hpp"
#include "core/common.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <sys/stat.h>
using tc::z_image::CoreMLGeneration;
namespace fs = std::filesystem;
void check(bool ok) { if (!ok) throw std::runtime_error("test assertion"); }
template<class F> void rejects(F f) { bool threw=false; try { f(); } catch (const std::exception &) { threw=true; } check(threw); }
void put(const fs::path &p, const std::string &s) { std::ofstream(p, std::ios::binary) << s; }
std::string get(const fs::path &p) { std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
int main(int argc, char **argv) {
    check(argc == 2 || argc == 3);
    fs::path root(argv[1]), src=root/"source", dest=root/"managed";
    fs::create_directories(src/"model.mlmodelc"/"empty"); fs::create_directory(dest);
    put(src/"manifest.json", "manifest");
    put(src/"model.mlmodelc"/"weights.bin", std::string(2*1024*1024+17, 'w'));
    auto first=CoreMLGeneration::import_tree(src,dest);
    auto first_path=first->root();
    check(first->copied_bytes()==8+2*1024*1024+17);
    check(get(first_path/"model.mlmodelc"/"weights.bin")==get(src/"model.mlmodelc"/"weights.bin"));
    check((fs::status(first_path/"manifest.json").permissions() & fs::perms::owner_write)==fs::perms::none);
    auto second_source=root/"copy"; fs::copy(src,second_source,fs::copy_options::recursive);
    auto second=CoreMLGeneration::import_tree(second_source,dest);
    check(first->content_digest()==second->content_digest()); check(first->root()!=second->root());
    put(src/"manifest.json", "updated manifest");
    first->revalidate(); check(get(first_path/"manifest.json")=="manifest");
    auto updated=CoreMLGeneration::import_tree(src,dest);
    check(updated->content_digest()!=first->content_digest()); first->revalidate();
    std::weak_ptr<const CoreMLGeneration> weak=first;
    auto retained=first; first.reset(); check(!weak.expired()); check(fs::exists(first_path));
    retained.reset(); check(weak.expired()); check(!fs::exists(first_path));
    auto count=std::distance(fs::directory_iterator(dest),fs::directory_iterator());
    std::atomic<bool> stop{true}; bool typed=false;
    try { CoreMLGeneration::import_tree(src,dest,&stop); } catch(const tc::Cancelled &) {typed=true;}
    check(typed); check(count==std::distance(fs::directory_iterator(dest),fs::directory_iterator()));
    fs::create_symlink(src/"manifest.json",src/"link"); rejects([&]{CoreMLGeneration::import_tree(src,dest);}); fs::remove(src/"link");
    fs::create_directory_symlink(src,root/"alias"); rejects([&]{CoreMLGeneration::import_tree(root/"alias",dest);});
    // Same-size mutation is rejected by generation metadata, independently of name inventory.
    auto file=second->root()/"manifest.json";
    fs::permissions(file,fs::perms::owner_read|fs::perms::owner_write); put(file,"mutated!");
    rejects([&]{second->revalidate();}); auto second_path=second->root(); second.reset(); check(!fs::exists(second_path));
    auto extra=updated->root()/"injected";
    fs::permissions(updated->root(),fs::perms::owner_all); put(extra,"new");
    rejects([&]{updated->revalidate();}); updated.reset();
    check(fs::is_empty(dest));
    fs::create_directory(root/"empty"); rejects([&]{CoreMLGeneration::import_tree(root/"empty",dest);});
    check(mkfifo((src/"fifo").c_str(),0600)==0); rejects([&]{CoreMLGeneration::import_tree(src,dest);}); fs::remove(src/"fifo");
    auto renamed=CoreMLGeneration::import_tree(src,dest);
    auto original=renamed->root(), moved=dest/"moved";
    fs::rename(original,moved); fs::create_directory(original); put(original/"keep","replacement");
    rejects([&]{renamed->revalidate();}); renamed.reset();
    check(get(original/"keep")=="replacement"); // Destructor must not delete a replacement tree.
    fs::remove_all(original);
    fs::permissions(moved,fs::perms::owner_all);
    for (auto &entry:fs::recursive_directory_iterator(moved))
        if (entry.is_directory()) fs::permissions(entry.path(),fs::perms::owner_all);
    fs::remove_all(moved);
    check(fs::is_empty(dest));
    if (argc == 3) {
        auto real=CoreMLGeneration::import_tree(argv[2],dest);
        real->revalidate();
        std::cout << "real_tree_content_digest=" << real->content_digest()
                  << " copied_bytes=" << real->copied_bytes() << "\n";
        auto path=real->root(); real.reset(); check(!fs::exists(path));
    }
    std::cout << "generation copy/content identity, update isolation, lifetime, cancellation, symlink and mutation rejection passed\n";
}
